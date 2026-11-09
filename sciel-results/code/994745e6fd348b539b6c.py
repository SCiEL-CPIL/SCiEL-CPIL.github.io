"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Max value needed is 1000000 + 24 = 1000024
    MAX_VAL = N + WINDOW

    # Compute smallest prime factor restricted to primes < 24
    # lp[m] will be the smallest prime p < 24 that divides m, or 0 if none.
    lp = np.zeros(MAX_VAL + 1, dtype=np.int32)
    for p in PRIMES:
        # Only set lp where it hasn't been set yet (i.e., no smaller prime divides m)
        indices = np.arange(p, MAX_VAL + 1, p)
        mask = lp[indices] == 0
        if mask.any():
            lp[indices[mask]] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    # sp[m] = product of all prime powers p^k where p < 24 and p^k divides m.
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    # We need to process in increasing order to ensure sp[m//p] is computed.
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p != 0:
            sp[m] = sp[m // p] * p
        # else sp[m] remains 1

    # Fast distinct count using a frequency array
    # sp values are products of prime powers from primes < 24
    # The maximum sp value for m <= 1000024 is bounded
    # Let's find the actual max sp value
    max_sp = int(np.max(sp))
    
    # Create a counter array for sp values
    # sp values range from 1 to max_sp
    # For a window of 24 values, we can use a small counter
    
    def count_distinct_fast(a, b):
        """Count distinct values in sp[a:b]"""
        window = sp[a:b]
        # For small windows, using a Python set on the numpy array converted to list
        # is often faster than np.unique for very small arrays
        # But let's try both and see
        return len(set(window.tolist()))

    # Alternative: precompute a hash map or use sorting
    # For 24 elements, sorting and counting is O(24 log 24) which is fine
    
    def count_distinct_sort(a, b):
        window = np.sort(sp[a:b])
        # Count transitions
        if len(window) <= 1:
            return len(window)
        changes = np.where(window[1:] != window[:-1])[0]
        return len(changes) + 1

    # Test which is faster
    # Let's just use the set approach which is simple and fast for 24 elements
    
    best_n = 1000
    best_distinct = float('inf')
    
    # Strategy: 
    # Pass 1: Scan with step 10
    # Pass 2: Refine around best with step 1 in a window
    # Pass 3: If needed, scan with step 2 in remaining regions
    
    # Pass 1
    step1 = 10
    for n in range(1000, N + 1, step1):
        if time.time() > deadline - 25:
            break
        dc = count_distinct_fast(n + 1, n + 25)
        if dc < best_distinct:
            best_distinct = dc
            best_n = n
            if best_distinct <= 5:
                break

    # Pass 2: Refine around best_n
    if time.time() < deadline - 10:
        ref_start = max(1000, best_n - 1000)
        ref_end = min(N, best_n + 1000)
        for n in range(ref_start, ref_end + 1):
            if time.time() > deadline - 5:
                break
            dc = count_distinct_fast(n + 1, n + 25)
            if dc < best_distinct:
                best_distinct = dc
                best_n = n
                if best_distinct <= 5:
                    break

    # Pass 3: Broader scan with step 5 if time permits
    if time.time() < deadline - 15 and best_distinct > 5:
        for n in range(1000, N + 1, 5):
            if time.time() > deadline - 5:
                break
            dc = count_distinct_fast(n + 1, n + 25)
            if dc < best_distinct:
                best_distinct = dc
                best_n = n
                if best_distinct <= 5:
                    break

    # Pass 4: Final refinement with step 1 around best_n if time permits
    if time.time() < deadline - 5 and best_distinct > 5:
        ref_start = max(1000, best_n - 2000)
        ref_end = min(N, best_n + 2000)
        for n in range(ref_start, ref_end + 1):
            if time.time() > deadline - 2:
                break
            dc = count_distinct_fast(n + 1, n + 25)
            if dc < best_distinct:
                best_distinct = dc
                best_n = n

    return best_n

# EVOLVE-BLOCK-END
