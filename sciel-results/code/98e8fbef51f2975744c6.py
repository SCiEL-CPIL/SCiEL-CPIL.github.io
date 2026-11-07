"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    MAX = 10000000
    
    # Build sigma using numpy additive sieve (fastest for this size)
    import numpy as np
    sigma_arr = np.zeros(MAX + 1, dtype=np.int64)
    for i in range(1, MAX + 1):
        sigma_arr[i::i] += i
    sigma = sigma_arr.tolist()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [11, 6]
    
    # Now search for best pair
    best_err_num = None
    best_err_den = None
    best_a = 11
    best_b = 6
    
    # a/b ≈ 3/2 means 2a ≈ 3b, i.e., a ≈ 1.5b
    # For each b, check a candidates near 1.5b
    # Error = |2a - 3b| / (2b)
    # For even b: min non-zero |2a - 3b| = 2 (when a = 3b/2 ± 1), error = 2/(2b) = 1/b
    # For odd b: min |2a - 3b| = 1 (when a = (3b±1)/2), error = 1/(2b)
    
    # Pass 1: tight window (check closest candidates first)
    for b in range(2, MAX + 1):
        if time.time() > deadline:
            break
        
        sb = sigma[b]
        if sb == 0:
            continue
        
        if b % 2 == 0:
            a_target = (3 * b) // 2
            # Check a_target ± 1, ± 2, ± 3, ± 4, ± 5
            for d in range(1, 6):
                for a_cand in (a_target - d, a_target + d):
                    if 1 < a_cand <= MAX:
                        if sigma[a_cand] == sb:
                            num = 2 * d  # |2*(a_target+d) - 3b| = 2d for even b
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
        else:
            a_lo = (3 * b - 1) // 2
            a_hi = (3 * b + 1) // 2
            # Check a_lo, a_hi, then a_lo±1, a_hi±1, etc.
            for d in range(0, 5):
                for a_cand in (a_lo - d, a_hi + d):
                    if 1 < a_cand <= MAX:
                        if sigma[a_cand] == sb:
                            num = 1 + 2 * d  # |2*(a_lo+d) - 3b| = 1+2d for odd b
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        if best_err_num is not None:
            return [int(best_a), int(best_b)]
        return [11, 6]
    
    # Pass 2: wider window for smaller b (where 1/(2b) is still relatively large)
    # For b < 100000, search wider windows
    WIDE_B_LIMIT = 100000
    WIDE_WINDOW = 50
    
    for b in range(2, min(WIDE_B_LIMIT, MAX + 1)):
        if time.time() > deadline:
            break
        
        sb = sigma[b]
        if sb == 0:
            continue
        
        if b % 2 == 0:
            a_target = (3 * b) // 2
            for d in range(6, WIDE_WINDOW + 1):
                for a_cand in (a_target - d, a_target + d):
                    if 1 < a_cand <= MAX:
                        if sigma[a_cand] == sb:
                            num = 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
        else:
            a_lo = (3 * b - 1) // 2
            a_hi = (3 * b + 1) // 2
            for d in range(5, WIDE_WINDOW + 1):
                for a_cand in (a_lo - d, a_hi + d):
                    if 1 < a_cand <= MAX:
                        if sigma[a_cand] == sb:
                            num = 1 + 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
    
    if best_err_num is not None:
        return [int(best_a), int(best_b)]
    else:
        return [11, 6]
# EVOLVE-BLOCK-END
