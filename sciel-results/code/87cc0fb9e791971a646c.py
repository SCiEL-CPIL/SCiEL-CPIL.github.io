import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    
    # Helper to check strict subset
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Group masks by size for efficient candidate generation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        masks_by_size[popcount[mask]].append(mask)
    
    # Greedy Initialization based on structural insight
    # Base: Empty, Full, all singletons, all co-singletons
    current_masks = [0, full_mask]
    used_masks = {0, full_mask}
    
    # Add all singletons (size 1)
    for i in range(n):
        mask = 1 << i
        current_masks.append(mask)
        used_masks.add(mask)
        
    # Add all co-singletons (size 11)
    for i in range(n):
        mask = full_mask ^ (1 << i)
        current_masks.append(mask)
        used_masks.add(mask)
        
    # Current size is 1 + 1 + 12 + 12 = 26
    # Need 14 more. Pick size 6 sets (middle of lattice)
    random.seed(42)
    available_size_6 = [mask for mask in masks_by_size[6] if mask not in used_masks]
    random.shuffle(available_size_6)
    
    for mask in available_size_6[:14]:
        current_masks.append(mask)
        used_masks.add(mask)
            
    # Pad if necessary (should be exactly 40 now)
    while len(current_masks) < m:
        for size in range(n + 1):
            for mask in masks_by_size[size]:
                if mask not in used_masks:
                    current_masks.append(mask)
                    used_masks.add(mask)
                    break
            else:
                continue
            break

    # Calculate initial score
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s

    best_score = calc_score(current_masks)
    best_solution = list(current_masks)

    # Local Search: Simulated Annealing with smart candidate generation
    # Key insight: prefer candidates that are comparable with many existing sets
    # Use size-biased generation: sizes near 0, 1, 11, 12 are more comparable
    
    def get_comparability_with_others(mask, masks_list, exclude_idx):
        """Count how many sets in masks_list (excluding exclude_idx) are comparable with mask"""
        count = 0
        for j in range(len(masks_list)):
            if j == exclude_idx:
                continue
            if is_comparable(mask, masks_list[j]):
                count += 1
        return count
    
    # Temperature schedule for simulated annealing
    T_initial = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    T = T_initial
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if time.time() > deadline - 2.0:
            break
            
        # Pick a random index to modify
        idx = random.randint(0, m - 1)
        old_mask = current_masks[idx]
        old_size = popcount[old_mask]
        
        # Generate candidate masks with bias towards useful sizes
        # Sizes 0,1,11,12 are very comparable; middle sizes less so
        # Weight sizes by how many comparables they'd likely have
        size_weights = [1.0] * (n + 1)
        # Boost sizes that are already well-represented or structurally useful
        size_weights[0] = 3.0
        size_weights[1] = 2.0
        size_weights[11] = 2.0
        size_weights[12] = 3.0
        # Slightly boost middle sizes for diversity
        for s in [5, 6, 7]:
            size_weights[s] = 1.5
        
        # Pick a size with weighted probability
        total_weight = sum(size_weights)
        r = random.random() * total_weight
        cum = 0
        chosen_size = 0
        for s in range(n + 1):
            cum += size_weights[s]
            if r <= cum:
                chosen_size = s
                break
        
        # Pick a random mask of that size not in used_masks
        candidates = masks_by_size[chosen_size]
        if not candidates:
            continue
            
        # Try a few random candidates from this size
        found_new = False
        for _ in range(5):
            new_mask = random.choice(candidates)
            if new_mask == old_mask or new_mask in used_masks:
                continue
            found_new = True
            break
        
        if not found_new:
            continue

        # Calculate delta efficiently
        old_comp_count = get_comparability_with_others(old_mask, current_masks, idx)
        new_comp_count = get_comparability_with_others(new_mask, current_masks, idx)
        delta = new_comp_count - old_comp_count
        
        # Simulated annealing acceptance
        if delta >= 0:
            # Accept
            used_masks.remove(old_mask)
            used_masks.add(new_mask)
            current_masks[idx] = new_mask
            best_score += delta
            best_solution = list(current_masks)
            no_improve_count = 0
        else:
            # Accept with probability exp(delta / T)
            if random.random() < pow(2.718281828, delta / T):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                best_score += delta
                best_solution = list(current_masks)
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # If stuck, restart with higher temperature
        if no_improve_count > max_no_improve:
            T = T_initial * 0.5
            no_improve_count = 0
    
    # Final polish: try to improve by swapping pairs
    for _ in range(100):
        if time.time() > deadline - 1.0:
            break
        improved = False
        for i in range(m):
            if time.time() > deadline - 1.0:
                break
            for j in range(i + 1, m):
                if time.time() > deadline - 1.0:
                    break
                # Try swapping masks[i] and masks[j] with other unused masks
                old_i = current_masks[i]
                old_j = current_masks[j]
                
                # Try replacing one of them
                for idx in [i, j]:
                    old_mask = current_masks[idx]
                    # Try a few random replacements
                    for _ in range(3):
                        new_mask = random.randint(0, full_mask)
                        if new_mask == old_mask or new_mask in used_masks:
                            continue
                        
                        old_comp = get_comparability_with_others(old_mask, current_masks, idx)
                        new_comp = get_comparability_with_others(new_mask, current_masks, idx)
                        delta = new_comp - old_comp
                        
                        if delta > 0:
                            used_masks.remove(old_mask)
                            used_masks.add(new_mask)
                            current_masks[idx] = new_mask
                            best_score += delta
                            best_solution = list(current_masks)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
