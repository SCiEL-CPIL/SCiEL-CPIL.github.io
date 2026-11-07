import itertools
import random
import math

def construct():
    n = 12
    m = 40
    
    # 1. Generate all subsets and map to indices
    all_subsets = []
    for r in range(n+1):
        for combo in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(combo))
            
    num_subsets = len(all_subsets) # 4096
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Precompute comparability masks
    # comp_masks[i] is an integer with bit j set if all_subsets[i] is comparable to all_subsets[j]
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
        
    def get_score_from_indices(indices):
        mask = 0
        for i in indices:
            mask |= (1 << i)
        score = 0
        for i in indices:
            score += popcount(comp_masks[i] & mask)
        return score // 2

    # 3. Initialization: Chain-based
    # Create a chain: {}, {0}, {0,1}, ..., {0..11}
    chain_sets = []
    current = frozenset()
    for i in range(n+1):
        chain_sets.append(current)
        if i < n:
            current = current.union({i})
            
    # We have 13 sets in the chain. We need 40.
    # Fill the rest with random sets, avoiding duplicates.
    used_indices = {subset_to_idx[s] for s in chain_sets}
    family_indices = list(used_indices)
    
    # Shuffle the remaining pool to pick random ones? 
    # Or just pick random indices not in used_indices.
    available = set(range(num_subsets)) - used_indices
    available_list = list(available)
    random.shuffle(available_list)
    
    for idx in available_list:
        if len(family_indices) >= m:
            break
        family_indices.append(idx)
        
    # Ensure we have exactly m
    if len(family_indices) < m:
        # Fallback if chain logic failed (it shouldn't)
        for idx in range(num_subsets):
            if idx not in family_indices:
                family_indices.append(idx)
            if len(family_indices) == m:
                break

    current_family = family_indices
    current_mask = 0
    for i in current_family:
        current_mask |= (1 << i)
    current_score = get_score_from_indices(current_family)
    
    best_family = list(current_family)
    best_score = current_score
    
    # 4. Simulated Annealing
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    T = T_start
    
    # Number of iterations
    # We have 60 seconds. 
    # Each iteration: 
    # 1. Pick random position (O(1))
    # 2. Pick random new index not in family (O(1) if we have a list of available, but maintaining it is costly. 
    #    Instead, pick random index from 0..4095, if in family, retry. Expected retries are low if family is small relative to 4096? 
    #    40/4096 is small, so collision probability is low. ~1% chance of collision. 
    # 3. Calculate delta score.
    #    Remove old, add new.
    #    Old contribution: popcount(comp_masks[old] & (mask ^ (1<<old)))
    #    New contribution: popcount(comp_masks[new] & (mask ^ (1<<old)))
    #    Delta = New - Old.
    #    This is 2 popcounts.
    # 4. Update state if accepted.
    
    # Estimate iterations: 
    # 100,000 iterations should be fast in Python if optimized.
    # Let's aim for 200,000 iterations.
    
    max_iterations = 200000
    iterations_per_T = max_iterations // 100 # Roughly
    
    for step in range(max_iterations):
        if T < T_end:
            break
            
        # Pick a random position to change
        pos = random.randint(0, m - 1)
        old_idx = current_family[pos]
        
        # Pick a random new index
        while True:
            new_idx = random.randint(0, num_subsets - 1)
            if new_idx != old_idx and not (current_mask & (1 << new_idx)):
                break
        
        # Calculate delta
        # mask without old_idx
        mask_without_old = current_mask ^ (1 << old_idx)
        
        old_comps = popcount(comp_masks[old_idx] & mask_without_old)
        new_comps = popcount(comp_masks[new_idx] & mask_without_old)
        
        delta = new_comps - old_comps
        
        if delta > 0:
            # Always accept improvement
            current_family[pos] = new_idx
            current_mask ^= (1 << old_idx)
            current_mask |= (1 << new_idx)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(current_family)
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                current_family[pos] = new_idx
                current_mask ^= (1 << old_idx)
                current_mask |= (1 << new_idx)
                current_score += delta
                
        # Cool down
        T *= cooling_rate
        
    # Convert back to lists of ints
    result = [sorted(list(all_subsets[idx])) for idx in best_family]
    return result
