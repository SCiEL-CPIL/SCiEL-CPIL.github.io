"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit

    # Precompute bit counts for all masks
    bit_counts = np.zeros(total_masks, dtype=np.int32)
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    # Precompute subset relation: sub[a][b] = 1 if a is subset of b (a & ~b == 0)
    # We need this for 4096x4096 which is 16M entries. Use numpy for speed.
    # Actually, checking (a | b) == b is equivalent to (a & ~b) == 0.
    # Let's precompute a lookup table for comparability?
    # Comparability: a is comparable to b if (a|b)==a or (a|b)==b.
    # Since sets are distinct in our selection, this means strict subset.
    
    # Instead of a full 4096x4096 matrix (16M bytes is fine, but computation might be slow in python loops),
    # let's use a smarter approach.
    
    # Helper to get list from mask
    def mask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]

    # --- Phase 1: Greedy Construction with Chain Bias ---
    # Start with a random chain to establish a strong backbone
    def generate_random_chain():
        perm = list(range(n))
        random.shuffle(perm)
        cur = 0
        chain = [0]
        for p in perm:
            cur |= (1 << p)
            chain.append(cur)
        return chain

    best_score = -1
    best_selection = []
    
    # Try multiple random initializations
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with a random chain
        selected_masks = generate_random_chain()
        selected_set = set(selected_masks)
        
        # Fill remaining slots greedily
        while len(selected_masks) < m:
            best_candidate = -1
            best_gain = -1
            
            # Candidate pool: all masks not in selected_set
            # To speed up, we can sample or iterate all 4096. 4096 is small.
            for cand in range(total_masks):
                if cand in selected_set:
                    continue
                
                # Calculate gain: number of comparable pairs with existing selected masks
                gain = 0
                for sel in selected_masks:
                    # Check comparability
                    if (sel | cand) == sel or (sel | cand) == cand:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            
            if best_candidate != -1:
                selected_masks.append(best_candidate)
                selected_set.add(best_candidate)
            else:
                # Should not happen as we only have 40 out of 4096
                break

        # Calculate initial score
        current_score = 0
        for i in range(len(selected_masks)):
            for j in range(i + 1, len(selected_masks)):
                if (selected_masks[i] | selected_masks[j]) == selected_masks[i] or \
                   (selected_masks[i] | selected_masks[j]) == selected_masks[j]:
                    current_score += 1
        
        if current_score > best_score:
            best_score = current_score
            best_selection = list(selected_masks)
        
        # --- Phase 2: Local Search (1-swap) ---
        # Try to improve by swapping one mask with another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(selected_masks)):
                if time.time() >= deadline:
                    break
                
                mask_to_remove = selected_masks[idx]
                
                # Find best replacement
                best_swap_gain = 0
                best_swap_mask = -1
                
                for cand in range(total_masks):
                    if cand in selected_set or cand == mask_to_remove:
                        continue
                    
                    # Calculate delta score
                    # Loss: comparabilities between mask_to_remove and other selected (excluding itself)
                    # Gain: comparabilities between cand and other selected (excluding itself)
                    delta = 0
                    for j in range(len(selected_masks)):
                        if j == idx:
                            continue
                        other = selected_masks[j]
                        
                        # Check if other is comparable to mask_to_remove
                        if (other | mask_to_remove) == other or (other | mask_to_remove) == mask_to_remove:
                            delta -= 1
                        
                        # Check if other is comparable to cand
                        if (other | cand) == other or (other | cand) == cand:
                            delta += 1
                    
                    if delta > best_swap_gain:
                        best_swap_gain = delta
                        best_swap_mask = cand
                
                if best_swap_gain > 0 and best_swap_mask != -1:
                    # Perform swap
                    selected_set.remove(mask_to_remove)
                    selected_set.add(best_swap_mask)
                    selected_masks[idx] = best_swap_mask
                    current_score += best_swap_gain
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected_masks)
                    
                    # Restart the sweep after a swap to find further improvements
                    break

    # Convert best_selection to lists of integers
    result = [mask_to_list(mask) for mask in best_selection]
    
    # Ensure types are correct: list of lists of int
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
# EVOLVE-BLOCK-END
