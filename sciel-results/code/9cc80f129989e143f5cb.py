import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    all_sets = list(itertools.combinations(range(n), 4))
    masks = []
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << x)
        masks.append(m)
    num_sets = len(all_sets)
    
    # Precompute req and forb for all pairs
    # req[i][j] = masks[i] & masks[j]
    # forb[i][j] = (masks[i] ^ req[i][j]) | (masks[j] ^ req[i][j])
    # Note: masks[i] ^ req[i][j] is masks[i] & ~req[i][j]
    
    req = [[0] * num_sets for _ in range(num_sets)]
    forb = [[0] * num_sets for _ in range(num_sets)]
    
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            r = masks[i] & masks[j]
            req[i][j] = r
            req[j][i] = r
            f = (masks[i] ^ r) | (masks[j] ^ r)
            forb[i][j] = f
            forb[j][i] = f

    def greedy_restart(seed_val):
        random.seed(seed_val)
        remaining = list(range(num_sets))
        random.shuffle(remaining)
        
        family = []
        conflict_count = [0] * num_sets
        in_family = [False] * num_sets
        
        # We need to update conflict_count for remaining candidates
        # when we add a new set.
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Find candidate with min conflict_count
            # We can just scan remaining
            best_cand = -1
            min_conf = float('inf')
            
            for c in remaining:
                if conflict_count[c] < min_conf:
                    min_conf = conflict_count[c]
                    best_cand = c
                    if min_conf == 0:
                        break
            
            if best_cand == -1 or min_conf > 0:
                break
                
            # Add best_cand to family
            family.append(best_cand)
            in_family[best_cand] = True
            remaining.remove(best_cand)
            
            # Update conflict counts for remaining candidates
            # For each existing set j in family (excluding best_cand? No, including pairs with best_cand)
            # Actually, we need to check pairs (j, best_cand) for all j in family[:-1]
            # And pairs (j, k) for j, k in family[:-1] were already accounted for?
            # No, conflict_count[c] counts the number of pairs (a, b) in family such that c conflicts with (a, b).
            # When we add best_cand, new pairs are (j, best_cand) for all j in family[:-1].
            # So for each remaining candidate c, we check if c conflicts with (j, best_cand).
            
            new_set = best_cand
            for j in family[:-1]:
                # Check if c conflicts with pair (j, new_set)
                r = req[j][new_set]
                f = forb[j][new_set]
                for c in remaining:
                    mc = masks[c]
                    if (mc & r) == r and (mc & f) == 0:
                        conflict_count[c] += 1

        return family

    best_family = []
    
    # Run multiple restarts
    # We can also try different initial shuffles
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        family = greedy_restart(r * 7919 + 42)
        
        if len(family) > len(best_family):
            best_family = family

    result = [all_sets[i] for i in best_family]
    return result
