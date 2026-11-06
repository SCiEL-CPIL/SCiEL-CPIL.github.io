import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
    
    # Precompute compatibility as bitmasks for fast lookup
    compat_mask = [0] * n_triples
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            if (mi & mj).bit_count() <= 1:
                compat_mask[i] |= (1 << j)
                compat_mask[j] |= (1 << i)
    
    def is_compatible(idx, T_set):
        """Check if idx is compatible with all triples in T_set (a set of indices)."""
        cm = compat_mask[idx]
        for t in T_set:
            if not (cm >> t) & 1:
                return False
        return True
    
    def get_alpha_fast(T_indices):
        """Compute independence number using bitmask DP over subsets of 16 points."""
        if not T_indices:
            return 16
        
        # Build a bitmask of all triples' masks
        # A subset S of points is independent if no triple is fully contained in S
        # We want the largest such S
        
        # Precompute which triples are "hit" by each point subset
        # For each subset of points (represented as 16-bit mask), check if any triple is fully contained
        
        # Use the fact that there are only 2^16 = 65536 possible subsets
        # For each subset, check if it's independent
        
        # First, build a set of triple masks
        triple_mask_set = set()
        for idx in T_indices:
            triple_mask_set.add(triple_masks[idx])
        
        # For each subset of points, check independence
        # A subset is independent if no triple mask is a subset of it
        
        # We can iterate from large to small and return the first independent set size
        # But checking all subsets is 65536 * |T| which might be slow
        
        # Better approach: iterate k from 16 down, check C(16,k) subsets
        # But use bit operations efficiently
        
        # Precompute: for each point, which triples contain it
        point_to_triples = [[] for _ in range(16)]
        for idx in T_indices:
            mask = triple_masks[idx]
            for p in range(16):
                if (mask >> p) & 1:
                    point_to_triples[p].append(idx)
        
        # Check subsets from k=16 down
        for k in range(16, 0, -1):
            for subset in itertools.combinations(range(16), k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                # Check if any triple is fully contained in smask
                independent = True
                for idx in T_indices:
                    tm = triple_masks[idx]
                    if (smask & tm) == tm:
                        independent = False
                        break
                
                if independent:
                    return k
        return 0
    
    def greedy_packing_pair_covering():
        """Build packing by greedily adding triples that cover the most uncovered point pairs."""
        T = []
        T_set = set()
        
        # Track which point pairs are covered
        # A pair {a,b} is covered if there's a triple containing both a and b
        pair_covered = [[False]*16 for _ in range(16)]
        
        while True:
            best_idx = -1
            best_score = -1
            
            for idx in range(n_triples):
                if idx in T_set:
                    continue
                if not is_compatible(idx, T_set):
                    continue
                
                # Count how many new point pairs this triple covers
                a, b, c = all_triples[idx]
                score = 0
                if not pair_covered[a][b]:
                    score += 1
                if not pair_covered[a][c]:
                    score += 1
                if not pair_covered[b][c]:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            T.append(best_idx)
            T_set.add(best_idx)
            a, b, c = all_triples[best_idx]
            pair_covered[a][b] = True
            pair_covered[b][a] = True
            pair_covered[a][c] = True
            pair_covered[c][a] = True
            pair_covered[b][c] = True
            pair_covered[c][b] = True
        
        return T
    
    def local_search(T, time_limit):
        """Improve T via various move operators."""
        current_T = T[:]
        current_T_set = set(current_T)
        current_alpha = get_alpha_fast(current_T)
        best_T = current_T[:]
        best_alpha = current_alpha
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding compatible triples
            for _ in range(10):
                idx = random.randint(0, n_triples - 1)
                if idx in current_T_set:
                    continue
                if not is_compatible(idx, current_T_set):
                    continue
                new_T = current_T + [idx]
                new_alpha = get_alpha_fast(new_T)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_T = new_T
                    current_T = new_T
                    current_T_set = set(new_T)
                    current_alpha = new_alpha
                    improved = True
                    break
                elif new_alpha == best_alpha and len(new_T) < len(current_T) + 5:
                    # Accept if same alpha but more triples (may help future moves)
                    current_T = new_T
                    current_T_set = set(new_T)
                    current_alpha = new_alpha
                    improved = True
                    break
            
            # Try 1-swap: remove one, add one
            if len(current_T) > 0:
                for _ in range(10):
                    remove_idx = random.choice(current_T)
                    add_idx = random.randint(0, n_triples - 1)
                    if add_idx == remove_idx or add_idx in current_T_set:
                        continue
                    
                    new_T_set = current_T_set - {remove_idx}
                    if not is_compatible(add_idx, new_T_set):
                        continue
                    
                    new_T = [t for t in current_T if t != remove_idx]
                    new_T.append(add_idx)
                    new_alpha = get_alpha_fast(new_T)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_T = new_T
                        current_T = new_T
                        current_T_set = set(new_T)
                        current_alpha = new_alpha
                        improved = True
                        break
            
            # Try 2-swap: remove two, add two
            if len(current_T) >= 2:
                for _ in range(5):
                    r1, r2 = random.sample(current_T, 2)
                    a1, a2 = random.sample(range(n_triples), 2)
                    if a1 in current_T_set or a2 in current_T_set or a1 == a2:
                        continue
                    
                    new_T_set = current_T_set - {r1, r2}
                    if not is_compatible(a1, new_T_set) or not is_compatible(a2, new_T_set):
                        continue
                    
                    new_T = [t for t in current_T if t not in (r1, r2)]
                    new_T.extend([a1, a2])
                    new_alpha = get_alpha_fast(new_T)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_T = new_T
                        current_T = new_T
                        current_T_set = set(new_T)
                        current_alpha = new_alpha
                        improved = True
                        break
            
            if not improved and random.random() < 0.05:
                # Occasionally restart
                fresh = greedy_packing_pair_covering()
                fresh_alpha = get_alpha_fast(fresh)
                if fresh_alpha < best_alpha:
                    best_alpha = fresh_alpha
                    best_T = fresh
                    current_T = fresh
                    current_T_set = set(fresh)
                    current_alpha = fresh_alpha
        
        return best_T, best_alpha
    
    best_alpha = 16
    best_system = []
    
    # Try the greedy pair-covering packing first
    T0 = greedy_packing_pair_covering()
    alpha0 = get_alpha_fast(T0)
    if alpha0 < best_alpha:
        best_alpha = alpha0
        best_system = T0[:]
    
    # Multiple restarts with local search
    while time.time() < deadline:
        # Randomized greedy packing
        order = list(range(n_triples))
        random.shuffle(order)
        
        T = []
        T_set = set()
        for idx in order:
            if is_compatible(idx, T_set):
                T.append(idx)
                T_set.add(idx)
        
        alpha = get_alpha_fast(T)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = T[:]
        
        # Local search
        if time.time() < deadline - 10:
            T, alpha = local_search(T, deadline - 5)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = T[:]
        
        if best_alpha <= 7:
            break
    
    if not best_system:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return [list(all_triples[i]) for i in best_system]
