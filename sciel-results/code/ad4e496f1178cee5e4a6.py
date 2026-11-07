import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    # Iterate over pairs (a, b) with a < b
    # a is value i+1, b is value j+1
    for i in range(128):
        if not (mask >> i) & 1:
            continue
        for j in range(i + 1, 128):
            if not (mask >> j) & 1:
                continue
            
            # Check if a+b is in S
            # a = i+1, b = j+1 => a+b = i+j+2
            # Bit index for a+b is (i+j+2)-1 = i+j+1
            if not (mask >> (i + j + 1)) & 1:
                continue
            
            # If a+b is in S, check if there exists c > b such that a+c in S and b+c in S
            # c is value k+1, where k > j
            # a+c = i+1 + k+1 = i+k+2 => Bit index i+k+1
            # b+c = j+1 + k+1 = j+k+2 => Bit index j+k+1
            
            # We need k such that:
            # 1. k > j
            # 2. Bit k in mask is 1 (c in S)
            # 3. Bit i+k+1 in mask is 1 (a+c in S)
            # 4. Bit j+k+1 in mask is 1 (b+c in S)
            
            # Let's form the intersection of conditions 2, 3, 4 for all k
            # Condition 3: Bit k in (mask >> (i+1)) is 1
            # Condition 4: Bit k in (mask >> (j+1)) is 1
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            
            # We only care about k > j
            # Mask for bits > j: all bits from j+1 to 127
            # (1 << 128) - 1 is all 1s.
            # We want to mask out bits 0..j
            # mask_gt_j = ((1 << 128) - 1) ^ ((1 << (j + 1)) - 1)
            # Or simply: candidates & ~((1 << (j + 1)) - 1)
            
            if candidates & ~((1 << (j + 1)) - 1):
                return False
    return True

def _greedy_construct_mask(deadline):
    """
    Greedy construction using bitmasks.
    """
    # Start with 65..128 => Bits 64..127
    best_mask = (1 << 128) - 1
    best_mask &= ~((1 << 64) - 1) # Clear bits 0..63 (values 1..64)
    
    # Try to add elements from 1..64
    for v in range(1, 65):
        if time.time() > deadline:
            break
        bit = 1 << (v - 1)
        trial_mask = best_mask | bit
        if _is_valid_mask(trial_mask):
            best_mask = trial_mask
    
    # Local search: try to swap elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and see if we can add two new ones
        for rem in range(1, 129):
            if time.time() > deadline:
                break
            rem_bit = 1 << (rem - 1)
            if not (best_mask & rem_bit):
                continue
            
            trial_base = best_mask & ~rem_bit
            # Try to add elements from 1..128 not in trial_base
            added = 0
            for add in range(1, 129):
                if add == rem:
                    continue
                add_bit = 1 << (add - 1)
                if trial_base & add_bit:
                    continue
                if time.time() > deadline:
                    break
                trial = trial_base | add_bit
                if _is_valid_mask(trial):
                    trial_base = trial
                    added += 1
            # Count bits to compare size
            if trial_base.bit_count() > best_mask.bit_count():
                best_mask = trial_base
                improved = True
    
    return best_mask

def _randomized_local_search_mask(deadline, seed=42):
    """
    Randomized local search with restarts using bitmasks.
    """
    random.seed(seed)
    # Start from 65..128
    best_mask = (1 << 128) - 1
    best_mask &= ~((1 << 64) - 1)
    best_size = best_mask.bit_count()
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start from a random valid set near the best
        current_mask = best_mask
        # Perturb: remove a random few and add random few
        for _ in range(3):
            if time.time() > deadline:
                break
            # Pick a random set bit to remove
            set_bits = [i for i in range(128) if (current_mask >> i) & 1]
            if not set_bits:
                break
            rem_idx = random.choice(set_bits)
            current_mask &= ~(1 << rem_idx)
            
            # Try to add a random element
            unset_bits = [i for i in range(128) if not (current_mask >> i) & 1]
            if unset_bits:
                add_idx = random.choice(unset_bits)
                trial_mask = current_mask | (1 << add_idx)
                if _is_valid_mask(trial_mask):
                    current_mask = trial_mask
        
        # Local search on current
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each element
            for add in range(1, 129):
                add_bit = 1 << (add - 1)
                if current_mask & add_bit:
                    continue
                if time.time() > deadline:
                    break
                trial_mask = current_mask | add_bit
                if _is_valid_mask(trial_mask):
                    current_mask = trial_mask
                    improved = True
                    if current_mask.bit_count() > best_size:
                        best_mask = current_mask
                        best_size = current_mask.bit_count()
            
            # Try removing one and adding two
            set_bits = [i for i in range(128) if (current_mask >> i) & 1]
            for rem_idx in set_bits:
                if time.time() > deadline:
                    break
                rem_bit = 1 << rem_idx
                trial_base = current_mask & ~rem_bit
                # Try to add elements
                added = 0
                for add in range(1, 129):
                    if add - 1 == rem_idx:
                        continue
                    add_bit = 1 << (add - 1)
                    if trial_base & add_bit:
                        continue
                    if time.time() > deadline:
                        break
                    trial = trial_base | add_bit
                    if _is_valid_mask(trial):
                        trial_base = trial
                        added += 1
                if trial_base.bit_count() > current_mask.bit_count():
                    current_mask = trial_base
                    improved = True
                    if current_mask.bit_count() > best_size:
                        best_mask = current_mask
                        best_size = current_mask.bit_count()
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    # Start with the seed solution (65..128)
    best_mask = (1 << 128) - 1
    best_mask &= ~((1 << 64) - 1)
    best_size = best_mask.bit_count()
    
    # Step 1: Greedy construction
    if time.time() < deadline - 20:
        greedy_result = _greedy_construct_mask(deadline - 20)
        if greedy_result.bit_count() > best_size:
            best_mask = greedy_result
            best_size = best_mask.bit_count()
    
    # Step 2: Randomized local search
    if time.time() < deadline - 5:
        rls_result = _randomized_local_search_mask(deadline - 5)
        if rls_result.bit_count() > best_size:
            best_mask = rls_result
            best_size = best_mask.bit_count()
    
    # Final check: make sure the result is valid
    if not _is_valid_mask(best_mask):
        # Fallback to seed
        best_mask = (1 << 128) - 1
        best_mask &= ~((1 << 64) - 1)
    
    # Convert mask to list of ints
    result = []
    for i in range(128):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
