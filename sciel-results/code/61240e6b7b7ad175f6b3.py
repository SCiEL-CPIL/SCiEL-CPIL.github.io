import time
import random
import itertools

# Precompute all triples and compatibility
N = 16
ALL_TRIPLES = []
TRIPLE_MASKS = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append((a, b, c))
            TRIPLE_MASKS.append(mask)

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute compatibility: compat[i] is a set of indices j such that triple i and j are compatible
COMPAT = [set() for _ in range(NUM_TRIPLES)]
for i in range(NUM_TRIPLES):
    for j in range(i + 1, NUM_TRIPLES):
        # Compatible if intersection size <= 1
        # Intersection size is popcount(mask_i & mask_j)
        inter = TRIPLE_MASKS[i] & TRIPLE_MASKS[j]
        if bin(inter).count('1') <= 1:
            COMPAT[i].add(j)
            COMPAT[j].add(i)

# Precompute point counts for each triple
POINT_COUNT = [0] * N
for tm in TRIPLE_MASKS:
    for p in range(N):
        if tm & (1 << p):
            POINT_COUNT[p] += 1

def calc_alpha(edge_indices):
    """
    Calculate the independence number of the hypergraph defined by edge_indices.
    Returns the size of the largest independent set.
    Uses full subset enumeration over 2^16 = 65536 subsets.
    """
    if not edge_indices:
        return N
    
    edge_masks = [TRIPLE_MASKS[i] for i in edge_indices]
    num_edges = len(edge_masks)
    
    # For each subset (as bitmask), check if it's independent
    # A subset S is independent if no edge is fully contained in S
    # i.e., for all edges e: (e_mask & S_mask) != e_mask
    # Equivalently: for all edges e: (e_mask | S_mask) == S_mask is FALSE
    # i.e., (e_mask & ~S_mask) != 0 for all edges e
    
    # We want the max popcount(S) such that S is independent
    # Iterate all 2^16 subsets
    best = 0
    for s_mask in range(1 << N):
        # Quick check: if popcount(s_mask) <= best, skip
        pc = bin(s_mask).count('1')
        if pc <= best:
            continue
        
        # Check if s_mask is independent
        is_indep = True
        for em in edge_masks:
            if (em & ~s_mask) == 0:
                # Edge em is fully contained in s_mask
                is_indep = False
                break
        
        if is_indep:
            best = pc
    
    return best

def greedy_construct():
    """
    Construct a maximal linear system using a greedy randomized approach.
    Returns list of edge indices.
    """
    order = list(range(NUM_TRIPLES))
    random.shuffle(order)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        # Check if idx is compatible with all selected
        compatible = True
        for s_idx in selected:
            if s_idx not in COMPAT[idx]:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
            selected_set.add(idx)
            
    return selected

def find_large_independent_set(edge_indices, target_size):
    """
    Find an independent set of size at least target_size, or None if none exists.
    Returns the set as a bitmask.
    """
    if not edge_indices:
        # All points are independent
        return (1 << N) - 1
    
    edge_masks = [TRIPLE_MASKS[i] for i in edge_indices]
    
    # Use recursive backtracking to find an independent set of size target_size
    # We'll try to build it greedily with pruning
    
    def is_independent(s_mask):
        for em in edge_masks:
            if (em & ~s_mask) == 0:
                return False
        return True
    
    # Try all subsets of size target_size using combinations
    # But that's expensive. Instead, use a heuristic: 
    # Start with all points, remove points that are in edges
    
    # Better: use the fact that we just need to check if alpha >= target_size
    # We can do this by checking all C(16, target_size) subsets
    # For target_size >= 8, C(16,8)=12870, C(16,9)=11440, etc.
    # This is feasible for target_size >= 5 or so
    
    if target_size > 8:
        # Check from target_size down
        for k in range(target_size, 7, -1):
            for subset in itertools.combinations(range(N), k):
                s_mask = 0
                for v in subset:
                    s_mask |= (1 << v)
                if is_independent(s_mask):
                    return s_mask
        return None
    else:
        # For smaller target sizes, just return a large set and verify
        # Actually, if target_size <= 4, it's almost always possible
        # Just return all points and check
        s_mask = (1 << N) - 1
        if is_independent(s_mask):
            return s_mask
        # Try removing one point at a time
        for p in range(N):
            s_mask = (1 << N) - 1 - (1 << p)
            if is_independent(s_mask):
                return s_mask
        # Try removing two points
        for p1 in range(N):
            for p2 in range(p1 + 1, N):
                s_mask = (1 << N) - 1 - (1 << p1) - (1 << p2)
                if is_independent(s_mask):
                    return s_mask
        return None

def local_search(edge_indices, max_iters=1000, deadline=None):
    """
    Try to improve the independence number by swapping edges.
    """
    current_edges = list(edge_indices)
    current_alpha = calc_alpha(current_edges)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if len(current_edges) < 2:
            break
            
        # Pick a random edge to remove
        rem_idx = random.choice(range(len(current_edges)))
        rem_edge = current_edges[rem_idx]
        
        # Remove it
        temp_edges = current_edges[:rem_idx] + current_edges[rem_idx+1:]
        temp_set = set(temp_edges)
        
        # Find a compatible edge to add that reduces alpha
        added = False
        # Try several random candidates
        for _ in range(20):
            cand_idx = random.randint(0, NUM_TRIPLES - 1)
            if cand_idx in temp_set:
                continue
            
            # Check compatibility
            ok = True
            for s_idx in temp_edges:
                if s_idx not in COMPAT[cand_idx]:
                    ok = False
                    break
            
            if ok:
                new_edges = temp_edges + [cand_idx]
                new_alpha = calc_alpha(new_edges)
                
                if new_alpha < current_alpha:
                    current_edges = new_edges
                    current_alpha = new_alpha
                    added = True
                    break
                elif new_alpha == current_alpha and random.random() < 0.15:
                    # Accept equal to explore
                    current_edges = new_edges
                    added = True
                    break
        
        if not added:
            # Try 2-swap: remove 2 edges, add 2 edges
            if len(current_edges) >= 4:
                rem1 = random.choice(range(len(current_edges)))
                rem2 = random.choice(range(len(current_edges) - 1))
                if rem2 >= rem1:
                    rem2 += 1
                temp_edges = current_edges[:rem1] + current_edges[rem1+1:rem2] + current_edges[rem2+1:]
                temp_set = set(temp_edges)
                
                for _ in range(10):
                    c1 = random.randint(0, NUM_TRIPLES - 1)
                    c2 = random.randint(0, NUM_TRIPLES - 1)
                    if c1 in temp_set or c2 in temp_set or c1 == c2:
                        continue
                    
                    ok1 = all(c1 in COMPAT[s_idx] for s_idx in temp_edges)
                    ok2 = all(c2 in COMPAT[s_idx] for s_idx in temp_edges)
                    if not ok1 or not ok2:
                        continue
                    # Check c1 and c2 are compatible with each other
                    if c2 not in COMPAT[c1]:
                        continue
                    
                    new_edges = temp_edges + [c1, c2]
                    new_alpha = calc_alpha(new_edges)
                    
                    if new_alpha < current_alpha:
                        current_edges = new_edges
                        current_alpha = new_alpha
                        added = True
                        break
                    elif new_alpha == current_alpha and random.random() < 0.1:
                        current_edges = new_edges
                        added = True
                        break
    
    return current_edges, current_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin (150s limit, 35s buffer)
    
    best_alpha = N
    best_system = []
    
    # Run multiple greedy + local search iterations
    iteration = 0
    while time.time() < deadline:
        # Generate a new greedy system
        sys = greedy_construct()
        
        # Local search to improve
        remaining_time = deadline - time.time()
        max_iters = min(2000, max(100, int(remaining_time * 50)))
        sys, alpha = local_search(sys, max_iters=max_iters, deadline=deadline)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = sys
        
        iteration += 1
        
        # If we hit a very low alpha, we might break early
        if best_alpha <= 4:
            break
            
    # Final polish: try to reduce alpha further with targeted edge additions
    if best_system and time.time() < deadline - 5:
        # Try adding edges that break large independent sets
        current = list(best_system)
        cur_alpha = calc_alpha(current)
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Find a large independent set
            indep_set = find_large_independent_set(current, cur_alpha)
            if indep_set is None:
                break
            
            # Try to add an edge that intersects this independent set in all 3 points
            # i.e., find a triple entirely within indep_set that's compatible with all current edges
            points_in_set = [p for p in range(N) if indep_set & (1 << p)]
            
            improved = False
            for a in points_in_set:
                for b in points_in_set:
                    if b <= a:
                        continue
                    for c in points_in_set:
                        if c <= b:
                            continue
                        # Find the triple index
                        tri = (a, b, c)
                        tri_idx = None
                        for i, t in enumerate(ALL_TRIPLES):
                            if t == tri:
                                tri_idx = i
                                break
                        if tri_idx is None:
                            continue
                        if tri_idx in set(current):
                            continue
                        
                        # Check compatibility with all current edges
                        ok = True
                        for s_idx in current:
                            if s_idx not in COMPAT[tri_idx]:
                                ok = False
                                break
                        
                        if ok:
                            new_edges = current + [tri_idx]
                            new_alpha = calc_alpha(new_edges)
                            if new_alpha < cur_alpha:
                                current = new_edges
                                cur_alpha = new_alpha
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_system = current
    
    # Convert indices back to triples
    return [ALL_TRIPLES[i] for i in best_system]
