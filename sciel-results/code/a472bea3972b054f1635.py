# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    all_ones = total_masks - 1  # 0xFFF
    
    # Precompute comparability bitsets
    # comp_bitset[i] is a bitmask of size total_masks where bit j is set if i and j are comparable (strict subset/superset)
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
        sub = (i - 1) & i
        while True:
            ci |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        
        # Enumerate all proper supermasks of i
        comp = all_ones ^ i
        if comp != 0:
            sub = (comp - 1) & comp
            while True:
                s = i | sub
                if s != i:
                    ci |= (1 << s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        
        comp_bitset[i] = ci

    def greedy_hill_climb(initial_selection=None):
        """
        Runs greedy construction if initial_selection is None, otherwise starts from it.
        Then performs steepest-ascent local search.
        Returns (score, selected_masks).
        """
        if initial_selection is None:
            # Greedy Construction
            selected = []
            selected_mask = 0
            # Start with empty set
            selected.append(0)
            selected_mask |= (1 << 0)
            
            for _ in range(m - 1):
                best_mask = -1
                best_score = -1
                for k in range(total_masks):
                    if (selected_mask >> k) & 1:
                        continue
                    score = (comp_bitset[k] & selected_mask).bit_count()
                    if score > best_score:
                        best_score = score
                        best_mask = k
                selected.append(best_mask)
                selected_mask |= (1 << best_mask)
        else:
            selected = list(initial_selection)
            selected_mask = 0
            for mask in selected:
                selected_mask |= (1 << mask)
        
        # Calculate initial score
        current_score = 0
        for i in range(m):
            current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
        current_score //= 2
        
        best_score = current_score
        best_selected = list(selected)
        
        # Local Search: Steepest Ascent
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 2.0:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                best_new_mask = -1
                best_delta = 0
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = new_mask
                
                if best_new_mask != -1 and best_delta > 0:
                    selected[idx] = best_new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    improved = True
                    break # Restart scan from index 0
                    
        return best_score, best_selected

    # Initial run
    best_score, best_selected = greedy_hill_climb(None)
    
    # Multi-start / Perturbation
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Perturb the best solution
        # Randomly swap 1 or 2 elements with random masks
        perturbed = list(best_selected)
        num_swaps = random.randint(1, 2)
        for _ in range(num_swaps):
            idx = random.randint(0, m - 1)
            new_mask = random.randint(0, total_masks - 1)
            # Ensure new_mask is not already in the set (except at idx)
            # We need to check if new_mask is in perturbed at a position != idx
            # If it is, we might create a duplicate, which is invalid.
            # So we must pick a new_mask that is not in the set.
            # Simple check:
            set_of_masks = set(perturbed)
            set_of_masks.remove(perturbed[idx])
            if new_mask in set_of_masks:
                # Retry or pick another
                # Just skip this swap
                continue
            perturbed[idx] = new_mask
        
        # Run hill climb from perturbed state
        score, selected = greedy_hill_climb(perturbed)
        if score > best_score:
            best_score = score
            best_selected = selected

    # Final polish on the absolute best found
    # Just in case the last perturbation led to a good start but we didn't finish the climb?
    # greedy_hill_climb runs until no improvement or timeout.
    # So best_selected is already locally optimal.
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure proper types
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
# EVOLVE-BLOCK-END
