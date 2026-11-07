"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Known good baseline: L = {16..31}, H = {64..128}, size 81
    baseline = set(range(16, 32)) | set(range(64, 129))
    best_set = baseline
    best_size = len(baseline)
    
    # Try different K values for the H = {K..128} structure
    for K in range(55, 75):
        if time.time() > deadline:
            break
        
        H = set(range(K, 129))
        max_L = (K - 1) // 2
        if max_L < 1:
            continue
        
        # Largest sum-free subset of {1, ..., max_L}
        start_L = max_L // 2 + 1
        L = set(range(start_L, max_L + 1))
        S = L | H
        
        if is_valid_fast(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Now try to improve via local search: add/remove elements
    # Convert to bitmask for fast operations
    best_mask = _set_to_mask(best_set)
    
    # Greedy add: try adding each element from 1..128
    for v in range(1, 129):
        if time.time() > deadline:
            break
        bit = 1 << (v - 1)
        if best_mask & bit:
            continue
        trial_mask = best_mask | bit
        if _is_valid_mask(trial_mask):
            best_mask = trial_mask
    
    best_size = best_mask.bit_count()
    
    # Local search: try swaps (remove one, add another)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for rem in range(1, 129):
            if time.time() > deadline:
                break
            rem_bit = 1 << (rem - 1)
            if not (best_mask & rem_bit):
                continue
            trial_base = best_mask & ~rem_bit
            for add in range(1, 129):
                if add == rem:
                    continue
                add_bit = 1 << (add - 1)
                if trial_base & add_bit:
                    continue
                trial = trial_base | add_bit
                if _is_valid_mask(trial):
                    if trial.bit_count() > best_size:
                        best_mask = trial
                        best_size = trial.bit_count()
                        improved = True
    
    # Try multi-swap: remove two, add two
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for rem1 in range(1, 129):
                if time.time() > deadline:
                    break
                rem1_bit = 1 << (rem1 - 1)
                if not (best_mask & rem1_bit):
                    continue
                for rem2 in range(rem1 + 1, 129):
                    if time.time() > deadline:
                        break
                    rem2_bit = 1 << (rem2 - 1)
                    if not (best_mask & rem2_bit):
                        continue
                    trial_base = best_mask & ~rem1_bit & ~rem2_bit
                    for add1 in range(1, 129):
                        if time.time() > deadline:
                            break
                        if add1 == rem1 or add1 == rem2:
                            continue
                        add1_bit = 1 << (add1 - 1)
                        if trial_base & add1_bit:
                            continue
                        for add2 in range(add1 + 1, 129):
                            if add2 == rem1 or add2 == rem2:
                                continue
                            add2_bit = 1 << (add2 - 1)
                            if trial_base & add2_bit:
                                continue
                            trial = trial_base | add1_bit | add2_bit
                            if _is_valid_mask(trial):
                                if trial.bit_count() > best_size:
                                    best_mask = trial
                                    best_size = trial.bit_count()
                                    improved = True
    
    return _mask_to_set(best_mask)

def _set_to_mask(S):
    """Convert a set of ints (1..128) to a bitmask."""
    mask = 0
    for v in S:
        mask |= (1 << (v - 1))
    return mask

def _mask_to_set(mask):
    """Convert a bitmask to a list of ints (1..128)."""
    result = []
    for i in range(128):
        if (mask >> i) & 1:
            result.append(i + 1)
    return result

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    for i in range(128):
        if not (mask >> i) & 1:
            continue
        for j in range(i + 1, 128):
            if not (mask >> j) & 1:
                continue
            
            # a = i+1, b = j+1 => a+b = i+j+2, bit index = i+j+1
            if i + j + 1 >= 128:
                continue
            if not (mask >> (i + j + 1)) & 1:
                continue
            
            # Need c > b (k > j) such that a+c and b+c in S
            # a+c = i+1+k+1 = i+k+2 => bit index i+k+1
            # b+c = j+1+k+1 = j+k+2 => bit index j+k+1
            # c in S => bit k
            
            # candidates: bits k where mask has bit k, bit i+k+1, and bit j+k+1 all set
            # mask & (mask >> (i+1)) & (mask >> (j+1)) gives bits k where all three conditions hold
            # but we need to be careful about bit positions
            
            # For bit k to satisfy:
            # - mask has bit k (c in S)
            # - mask has bit i+k+1 (a+c in S) => (mask >> (i+1)) has bit k
            # - mask has bit j+k+1 (b+c in S) => (mask >> (j+1)) has bit k
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            
            # We only care about k > j (c > b)
            # Mask out bits 0..j
            if candidates & ~((1 << (j + 1)) - 1):
                return False
    return True

def is_valid_fast(S):
    """Fast validity check using bitmask."""
    mask = _set_to_mask(S)
    return _is_valid_mask(mask)

def is_valid(S):
    """Check if set S has no violating triples."""
    return is_valid_fast(S)

# EVOLVE-BLOCK-END
