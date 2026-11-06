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
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial set: random or heuristic?
    # Let's start with a random set of 32 numbers in 1..512.
    # Or maybe a structured start? The seed was range(1,33).
    # Let's try a few random restarts if time permits, but focus on one good SA run.
    
    random.seed(42) # For reproducibility in this specific run? No, better to be deterministic or just let it run.
    # The task says "Maximize". The evaluator runs it.
    
    # Heuristic initialization: Try to find a good starting point quickly.
    # Maybe start with a known good structure or just random.
    # Let's use a random start.
    current_set = set(random.sample(range(1, 513), 32))
    current_mask = 0
    for x in current_set:
        current_mask |= (1 << (x - 1))
    
    current_score = _fast_score_bitmask(current_mask)
    best_score = current_score
    best_mask = current_mask
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    no_improve = 0
    
    while time.time() < deadline:
        # Generate move: swap one element out, one in
        # Or remove one, add one.
        # Swap is better for maintaining size.
        
        # Pick an element to remove
        out_idx = random.randint(0, 31)
        # We need to map index to value. Since set is unsorted, let's keep a list.
        # It's easier to keep a list of the 32 numbers.
        
        # Let's refactor to use a list for the current state.
        pass # This approach is getting messy with mask vs list. Let's stick to list for moves, mask for scoring.

    # Let's restart the logic properly inside the function.
    
    # Reset for proper implementation
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
        
        # Pick a new value not in current_list
        # To avoid O(N) check, we can just pick random and retry, or maintain a boolean array.
        # Given 512 range, picking random and checking set membership is O(1).
        # But we need to ensure it's not already in the set.
        
        # Efficient way:
        # 1. Remove old_val from mask
        mask_wo_old = current_mask & ~(1 << (old_val - 1))
        
        # 2. Pick a candidate new_val
        # We want to bias towards values that might form progressions?
        # For now, random.
        # To pick a value not in the set:
        # We can generate a random number in 1..512. If it's in the set (and not the one we removed), retry.
        # Or, simpler: pick from the 480 available numbers.
        # Let's just use random.randint and check.
        
        while True:
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            # Check if new_val is in the current set (excluding old_val which is being removed)
            # If new_val is in the set, it's a duplicate, which is invalid.
            # We check if bit is set in mask_wo_old.
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
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Metropolis criterion
            # Accept with probability exp(delta / T)
            # delta is negative
            p = np.exp(delta / T)
            if random.random() < p:
                current_mask = new_mask
                current_score = new_score
                current_list[i] = new_val
                no_improve += 1
            else:
                no_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # If stuck for too long, maybe restart or reheat?
        # For now, just keep going.
        
    # Return the best list found
    return best_list

# EVOLVE-BLOCK-END
