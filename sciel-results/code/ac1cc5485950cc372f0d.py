"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def _mask_to_list(mask):
    """Convert a 16-bit mask to a sorted list of 1-based indices."""
    return [i + 1 for i in range(16) if (mask >> i) & 1]

def construct():
    # Generate all 4-subsets of [16] as 16-bit masks
    all_masks = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                for l in range(k + 1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_masks.append(mask)
    
    n_sets = len(all_masks)
    best_size = 0
    best_masks = []
    
    rng = random.Random(42)
    
    # Increase number of trials due to speed of bitmask operations
    # 2184 sets, greedy is O(N * |F|^2). With |F|~100, N=2184, ops ~ 2184 * 10000 = 22M per trial.
    # In Python, 22M simple int ops is fast (<0.1s). We can do many trials.
    for trial in range(500):
        order = list(range(n_sets))
        rng.shuffle(order)
        
        chosen_idx = []
        chosen_masks = []
        
        for idx in order:
            s = all_masks[idx]
            conflict_found = False
            
            # Check against all pairs in chosen
            # Optimization: If chosen is large, this is the bottleneck.
            # We can break early.
            for i in range(len(chosen_masks)):
                a = chosen_masks[i]
                ai = a & s
                for j in range(i + 1, len(chosen_masks)):
                    b = chosen_masks[j]
                    if (a & b) == ai == (b & s):
                        conflict_found = True
                        break
                if conflict_found:
                    break
            
            if not conflict_found:
                chosen_idx.append(idx)
                chosen_masks.append(s)
        
        if len(chosen_idx) > best_size:
            best_size = len(chosen_idx)
            best_masks = chosen_masks[:]
            
    # Convert best masks to the required format: list of sorted lists
    return [_mask_to_list(m) for m in best_masks]
# EVOLVE-BLOCK-END
