"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _score_bitmask(mask: int) -> int:
    """
    Count distinct d > 0 such that there exists a with a, a+d, a+2d in the set.
    mask is a 512-bit integer where bit i (0-indexed) represents value i+1.
    """
    count = 0
    for d in range(1, 257):  # max d is 255 (512-1-1... actually max a=1, a+2d<=512 => 2d<=511 => d<=255)
        # For difference d, we need a, a+d, a+2d all in set.
        # In bitmask terms: for some a (1-indexed), bits a-1, a+d-1, a+2d-1 are all set.
        # Let's shift the mask.
        # mask has bit i set if (i+1) is in set.
        # We want to check if there exists i such that bit i, bit i+d, bit i+2d are set.
        # i ranges from 0 to 512-1-2d.
        # mask & (mask >> d) & (mask >> 2*d)
        # If any bit is set in this AND, then there is a progression.
        # Note: shifting right by d moves bit i+d to position i.
        # So bit i in (mask >> d) corresponds to original bit i+d.
        # Bit i in (mask >> 2*d) corresponds to original bit i+2d.
        # So mask & (mask >> d) & (mask >> 2*d) has bit i set iff original bits i, i+d, i+2d are set.
        # We just need to check if this integer is non-zero.
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _fast_score_bitmask(mask: int) -> int:
    """
    Vectorized-ish scoring using precomputed shifts? 
    Actually, pure Python loop over 255 differences is quite fast.
    Let's benchmark mentally: 255 iterations, each doing 3 shifts and 2 ANDs and a check.
    This is ~2000 bit operations per score. Very fast in CPython big ints.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    random.seed(42)
    
    # Initial set: random
    current_list = random.sample(range(1, 513), 32)
    current_mask = 0
    for x in current_list:
        current_mask |= (1 << (x - 1))
    current_score = _fast_score_bitmask(current_mask)
    best_score = current_score
    best_mask = current_mask
    best_list = list(current_list)
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    while time.time() < deadline:
        # Move: Pick a random index in current_list to replace
        i = random.randint(0, 31)
        old_val = current_list[i]
        
        # 1. Remove old_val from mask
        mask_wo_old = current_mask & ~(1 << (old_val - 1))
        
        # 2. Pick a candidate new_val
        while True:
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            # Check if new_val is in the current set (excluding old_val)
            if (mask_wo_old >> (new_val - 1)) & 1:
                continue
            break
        
        # Apply move
        new_mask = mask_wo_old | (1 << (new_val - 1))
        new_score = _fast_score_bitmask(new_mask)
        
        delta = new_score - current_score
        
        if delta >= 0:
            # Accept
            current_mask = new_mask
            current_score = new_score
            current_list[i] = new_val
            
            if new_score > best_score:
                best_score = new_score
                best_mask = new_mask
                best_list = list(current_list)
        else:
            # Metropolis criterion
            p = np.exp(delta / T)
            if random.random() < p:
                current_mask = new_mask
                current_score = new_score
                current_list[i] = new_val
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    # Return the best list found
    return best_list

# EVOLVE-BLOCK-END
