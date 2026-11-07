import itertools
import random
import math

def construct():
    n = 12
    m = 40
    
    # 1. Generate all subsets
    all_subsets = []
    for r in range(n+1):
        for combo in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(combo))
            
    num_subsets = len(all_subsets) # 4096
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

    # Helper to compute score of a family given its mask and list
    def get_score(family_indices, family_mask):
        score = 0
        for i in family_indices:
            score += popcount(comp_masks[i] & family_mask)
        return score // 2

    # 3. Greedy Initialization
    family_indices = []
    family_set = set()
    
    init_sets = [frozenset(), frozenset(range(n))]
    family_indices = [subset_to_idx[s] for s in init_sets]
    family_set = set(family_indices)
    
    remaining_indices = set(range(num_subsets)) - family_set
    
    while len(family_indices) < m:
        best_idx = -1
        best_gain = -1
        for idx in remaining_indices:
            s = all_subsets[idx]
            gain = sum(1 for i in family_indices if all_subsets[idx] < all_subsets[i] or all_subsets[i] < all_subsets[idx])
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        family_indices.append(best_idx)
        family_set.add(best_idx)
        remaining_indices.discard(best_idx)
        
    best_score = get_score(family_indices, sum(1 << i for i in family_indices))
    best_family = list(family_indices)
    
    # 4. Simulated Annealing
    # We will run SA for a fixed number of steps or time
    # To ensure speed, we limit steps.
    
    def run_sa(initial_family, steps=50000, T_start=10.0, T_end=0.01):
        current_family = list(initial_family)
        current_mask = 0
        for i in current_family:
            current_mask |= (1 << i)
        current_score = get_score(current_family, current_mask)
        
        # Precompute list of indices for random selection?
        # We need to pick i from current_family and j from NOT current_family.
        # Picking j from NOT current_family is tricky if we don't maintain a set.
        # We can pick a random j from 0..4095 and reject if in mask.
        # Probability of rejection is 40/4096 ~ 1%, so this is fast.
        
        for step in range(steps):
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (step / steps)
            
            # Pick random i from family
            i_pos = random.randint(0, m - 1)
            i_idx = current_family[i_pos]
            
            # Pick random j not in family
            while True:
                j_idx = random.randint(0, num_subsets - 1)
                if not (current_mask & (1 << j_idx)):
                    break
            
            # Calculate delta score
            # Remove i, add j
            # Score change = (comps(j, F\{i}) - comps(i, F\{i}))
            # F\{i} mask = current_mask ^ (1 << i_idx)
            other_mask = current_mask ^ (1 << i_idx)
            
            comps_i = popcount(comp_masks[i_idx] & other_mask)
            comps_j = popcount(comp_masks[j_idx] & other_mask)
            
            delta = comps_j - comps_i
            
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept move
                current_family[i_pos] = j_idx
                current_mask ^= (1 << i_idx)
                current_mask |= (1 << j_idx)
                current_score += delta
                
        return current_score, current_family

    # Run multiple restarts
    for restart in range(5):
        if restart == 0:
            init = list(best_family)
        else:
            # Perturb best family
            init = list(best_family)
            for _ in range(10):
                i_pos = random.randint(0, m - 1)
                i_idx = init[i_pos]
                mask = 0
                for idx in init:
                    mask |= (1 << idx)
                while True:
                    j_idx = random.randint(0, num_subsets - 1)
                    if not (mask & (1 << j_idx)):
                        break
                init[i_pos] = j_idx
                
        score, family = run_sa(init, steps=20000, T_start=5.0, T_end=0.01)
        
        if score > best_score:
            best_score = score
            best_family = family
            
    # Convert back to lists of ints
    result = [sorted(list(all_subsets[idx])) for idx in best_family]
    return result
