import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    points = list(range(n))
    
    # All possible triples
    all_triples = list(combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    def calc_alpha(triples):
        """Compute independence number via branch-and-bound."""
        if not triples:
            return n
        
        # Build bitmask for each triple
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)
        
        # For each point, which triples contain it (as indices)
        point_in_triples = [[] for _ in range(n)]
        for ti, tm in enumerate(t_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_in_triples[p].append(ti)
        
        # Branch and bound
        best = [0]
        
        def is_independent(mask):
            for tm in t_masks:
                if (mask & tm) == tm:
                    return False
            return True
        
        def branch(remaining, current_mask, depth):
            if time.time() > deadline - 0.5:
                return
            if not remaining:
                cnt = bin(current_mask).count('1')
                if cnt > best[0]:
                    best[0] = cnt
                return
            
            # Upper bound: current + remaining
            ub = bin(current_mask).count('1') + len(remaining)
            if ub <= best[0]:
                return
            
            # Pick the point in remaining that appears in most triples
            best_point = -1
            best_count = -1
            for p in remaining:
                c = len(point_in_triples[p])
                if c > best_count:
                    best_count = c
                    best_point = p
            
            # Branch 1: include best_point
            new_mask = current_mask | (1 << best_point)
            # Check if still independent
            ok = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    ok = False
                    break
            if ok:
                new_remaining = [p for p in remaining if p != best_point]
                branch(new_remaining, new_mask, depth + 1)
            
            # Branch 2: exclude best_point
            new_remaining = [p for p in remaining if p != best_point]
            branch(new_remaining, current_mask, depth + 1)
        
        branch(list(range(n)), 0, 0)
        return best[0]
    
    def is_valid_addition(triple, used_pairs):
        """Check if triple can be added without pair conflicts."""
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = triple[i], triple[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    return False
        return True
    
    def add_pairs(triple, used_pairs):
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = triple[i], triple[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                used_pairs.add(key)
    
    def remove_pairs(triple, used_pairs):
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = triple[i], triple[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                used_pairs.discard(key)
    
    def get_max_independent_set(triples):
        """Find a maximum independent set (not just its size)."""
        if not triples:
            return list(range(n))
        
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)
        
        best_set = [None]
        best_size = [0]
        
        def is_independent(mask):
            for tm in t_masks:
                if (mask & tm) == tm:
                    return False
            return True
        
        def branch(remaining, current_mask):
            if time.time() > deadline - 0.5:
                return
            if not remaining:
                cnt = bin(current_mask).count('1')
                if cnt > best_size[0]:
                    best_size[0] = cnt
                    best_set[0] = [p for p in range(n) if current_mask & (1 << p)]
                return
            
            ub = bin(current_mask).count('1') + len(remaining)
            if ub <= best_size[0]:
                return
            
            # Pick point in remaining
            best_point = remaining[0]
            for p in remaining:
                if p > best_point:
                    best_point = p
            # Actually pick first for determinism in this helper
            
            # Include
            new_mask = current_mask | (1 << best_point)
            ok = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    ok = False
                    break
            if ok:
                new_remaining = [p for p in remaining if p != best_point]
                branch(new_remaining, new_mask)
            
            # Exclude
            new_remaining = [p for p in remaining if p != best_point]
            branch(new_remaining, current_mask)
        
        branch(list(range(n)), 0)
        return best_set[0] if best_set[0] else list(range(n))
    
    def local_search(triples, deadline):
        """Improve alpha by targeted moves."""
        if not triples:
            return triples, n
        
        current_alpha = calc_alpha(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        
        # Precompute used pairs
        used_pairs = set()
        for t in triples:
            add_pairs(t, used_pairs)
        
        # Candidate triples not in current
        current_set = set(triples)
        candidates = [t for t in all_triples if t not in current_set]
        
        # Simulated annealing with targeted moves
        temperature = 0.5
        cooling = 0.995
        current = triples[:]
        cur_alpha = current_alpha
        
        max_steps = 8000
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            
            if step % 50 == 0:
                temperature *= cooling
            
            # Get current independent set
            ind_set = get_max_independent_set(current)
            
            # Targeted move: try to break the independent set
            if ind_set and len(ind_set) > 4:
                # Find a triple that hits the most points in ind_set
                best_hit = -1
                best_hit_count = 0
                for t in candidates:
                    hit = sum(1 for p in t if p in ind_set)
                    if hit > best_hit_count:
                        best_hit_count = hit
                        best_hit = t
                
                if best_hit >= 0 and best_hit_count >= 2:
                    # Try to add best_hit by removing a conflicting triple
                    # Find which triples conflict with best_hit
                    conflicting = []
                    for t in current:
                        for i in range(3):
                            for j in range(i+1, 3):
                                p1, p2 = t[i], t[j]
                                key = (p1, p2) if p1 < p2 else (p2, p1)
                                for k in range(3):
                                    for l in range(k+1, 3):
                                        p3, p4 = best_hit[k], best_hit[l]
                                        key2 = (p3, p4) if p3 < p4 else (p4, p3)
                                        if key == key2:
                                            conflicting.append(t)
                                            break
                                    else:
                                        continue
                                    break
                            else:
                                continue
                            break
                    
                    if conflicting:
                        # Remove one conflicting triple and add best_hit
                        victim = random.choice(conflicting)
                        idx = current.index(victim)
                        new_triples = current[:idx] + current[idx+1:] + [best_hit]
                        new_alpha = calc_alpha(new_triples)
                        
                        delta = new_alpha - cur_alpha
                        if delta < 0:
                            current = new_triples
                            cur_alpha = new_alpha
                            if cur_alpha < best_alpha:
                                best_alpha = cur_alpha
                                best_triples = current[:]
                        else:
                            if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                                current = new_triples
                                cur_alpha = new_alpha
                        continue
            
            # Standard swap move
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            
            # Find compatible candidates
            temp_used = set()
            for t in current:
                if t == removed:
                    continue
                add_pairs(t, temp_used)
            
            valid_cands = []
            for t in candidates:
                if t == removed:
                    continue
                if is_valid_addition(t, temp_used):
                    valid_cands.append(t)
            
            if not valid_cands:
                continue
            
            added = random.choice(valid_cands)
            new_triples = current[:idx] + current[idx+1:] + [added]
            new_alpha = calc_alpha(new_triples)
            
            delta = new_alpha - cur_alpha
            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
            else:
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current = new_triples
                    cur_alpha = new_alpha
        
        return best_triples, best_alpha
    
    # Greedy construction
    def greedy_construct():
        triples = []
        used_pairs = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            if is_valid_addition(t, used_pairs):
                triples.append(t)
                add_pairs(t, used_pairs)
        return triples
    
    # Main: multiple restarts
    best_triples = []
    best_alpha = n
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        triples = greedy_construct()
        if not triples:
            continue
        
        triples, alpha = local_search(triples, deadline)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        if best_alpha <= 4:
            break
    
    # Use remaining time for more restarts
    while time.time() < deadline - 5:
        triples = greedy_construct()
        if not triples:
            continue
        triples, alpha = local_search(triples, deadline)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
    
    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    used_pairs = set()
    valid = True
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    valid = False
                    break
                used_pairs.add(key)
        if not valid:
            break
    if valid:
        print("Valid linear system")
