import random
import math
import itertools

def construct():
    n = 12
    m = 40
    
    # 1. Generate all subsets
    all_subsets = []
    for r in range(n+1):
        for combo in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(combo))
            
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Precompute comparability masks
    comp_masks = [0] * num_subsets
    for i in range(num_subsets):
        si = all_subsets[i]
        mask = 0
        for j in range(num_subsets):
            sj = all_subsets[j]
            if si < sj or sj < si:
                mask |= (1 << j)
        comp_masks[i] = mask
        
    def popcount(x):
        return bin(x).count('1')
        
    # 3. Simulated Annealing
    def random_init():
        indices = random.sample(range(num_subsets), m)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        return indices, mask

    def get_score(indices, mask):
        score = 0
        for i in indices:
            score += popcount(comp_masks[i] & mask)
        return score // 2

    best_indices, best_mask = random_init()
    best_score = get_score(best_indices, best_mask)
    
    def run_sa(start_indices, start_mask, start_score, iters, t_start):
        nonlocal best_indices, best_mask, best_score
        
        indices = list(start_indices)
        mask = start_mask
        score = start_score
        
        # Build available list
        in_family = [False] * num_subsets
        for i in indices:
            in_family[i] = True
        available = [i for i in range(num_subsets) if not in_family[i]]
        
        t = t_start
        cooling = 0.9999
        
        for _ in range(iters):
            # Pick random position in family
            pos = random.randint(0, m - 1)
            old_idx = indices[pos]
            
            # Pick random candidate from available
            avail_pos = random.randint(0, len(available) - 1)
            new_idx = available[avail_pos]
            
            # Calculate delta score
            mask_without_old = mask ^ (1 << old_idx)
            
            deg_old = popcount(comp_masks[old_idx] & mask_without_old)
            deg_new = popcount(comp_masks[new_idx] & mask_without_old)
            
            delta = deg_new - deg_old
            
            if delta > 0 or random.random() < math.exp(delta / t):
                # Accept move
                score += delta
                mask = mask_without_old | (1 << new_idx)
                indices[pos] = new_idx
                
                # Update available list
                available[avail_pos] = old_idx
                
                if score > best_score:
                    best_score = score
                    best_indices = list(indices)
                    best_mask = mask
            
            t *= cooling
            
        return best_indices, best_mask, best_score

    # Run multiple restarts
    # 20 restarts * 500k iters
    for restart in range(20):
        if restart == 0:
            curr_indices, curr_mask = random_init()
            curr_score = get_score(curr_indices, curr_mask)
        else:
            # Perturb best solution
            curr_indices = list(best_indices)
            curr_mask = best_mask
            curr_score = best_score
            
            # Randomly swap 10 elements
            for _ in range(10):
                pos = random.randint(0, m - 1)
                old_idx = curr_indices[pos]
                while True:
                    cand = random.randint(0, num_subsets - 1)
                    if not (curr_mask & (1 << cand)):
                        break
                curr_indices[pos] = cand
                curr_mask ^= (1 << old_idx)
                curr_mask |= (1 << cand)
            
            curr_score = get_score(curr_indices, curr_mask)
        
        best_indices, best_mask, best_score = run_sa(curr_indices, curr_mask, curr_score, 500000, 10.0)

    # Convert back to lists of ints
    result = [sorted(list(all_subsets[idx])) for idx in best_indices]
    return result
