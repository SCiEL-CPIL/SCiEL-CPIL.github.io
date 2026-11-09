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
        lp[p::p] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    # sp[m] = product of all prime powers p^k where p < 24 and p^k divides m.
    # This is equivalent to m / (m / sp[m]) where we remove all prime factors >= 24.
    # We can compute this iteratively.
    # sp[m] = sp[m // p] * p if p is the smallest prime factor < 24 of m.
    # If lp[m] == 0, then sp[m] = 1 (since no prime < 24 divides m, and sp[1]=1).
    # Wait, if m has no prime factors < 24, then sp[m] = 1.
    # If m has prime factors < 24, let p be the smallest.
    # m = p * k.
    # sp[m] = p * sp[k] ?
    # Let's check: m = 12 = 2 * 6. lp[12]=2. sp[12] should be 12.
    # sp[6] = 6. sp[12] = 2 * 6 = 12. Correct.
    # m = 10 = 2 * 5. sp[10] = 10. sp[5]=5. 2*5=10. Correct.
    # m = 25 = 5 * 5. lp[25]=5. sp[25] = 5 * sp[5] = 5 * 5 = 25. Correct.
    # m = 7 = 7 * 1. lp[7]=7. sp[7] = 7 * sp[1] = 7 * 1 = 7. Correct.
    # m = 29. lp[29]=0. sp[29] = 1. Correct.
    
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    # We need to process in increasing order to ensure sp[m//p] is computed.
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p != 0:
            sp[m] = sp[m // p] * p
        # else sp[m] remains 1

    # Coarse-to-fine search
    best_n = 1000
    best_distinct = float('inf')
    
    # Step 1: Coarse scan
    # Check n at intervals of COARSE_STEP
    COARSE_STEP = 100
    # We can optimize the distinct count calculation using a set on a numpy slice.
    # set(sp[n+1:n+25]) is fast for small windows.
    
    coarse_candidates = []
    
    for n in range(1000, N + 1, COARSE_STEP):
        if time.time() > deadline - 10:
            break
        # Extract window sp[n+1] to sp[n+24]
        window = sp[n+1:n+25]
        distinct_count = len(set(window.tolist()))
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
            coarse_candidates = [] # Reset if we found a new global best? 
                                   # Or keep accumulating? 
                                   # Actually, we just want the best.
        elif distinct_count == best_distinct:
            # If we find another with same count, we might want to check neighbors
            # but for now, we just record the best.
            pass
            
        # Heuristic: If we find a very low count (e.g., <= 6), we can break early?
        # The theoretical minimum is likely around 5-7.
        # If we find 5, it's probably optimal.
        if best_distinct <= 5:
            break

    # Step 2: Refine around the best candidate found in coarse scan
    # Also, we should refine around the initial best_n if it was set.
    # But coarse scan might have skipped the true minimum if it was between steps.
    # So we refine around every coarse candidate that was "good" (close to best_distinct).
    # Actually, simpler: just refine around the single best_n found so far.
    # But what if the true min is at n+50 and coarse step is 100?
    # We might have missed it if its score was slightly worse than the coarse hit.
    # Better: refine around the best coarse hit, AND maybe check a few nearby coarse hits.
    
    # Let's collect the top 5 coarse candidates
    # To do this properly, we'd need to store them. 
    # Let's just re-scan the neighborhood of the best_n found.
    # And also, to be safe, we can do a second pass with a smaller step 
    # in the region of the best coarse hit.
    
    # Refinement region: best_n +/- 200
    ref_start = max(1000, best_n - 200)
    ref_end = min(N, best_n + 200)
    
    for n in range(ref_start, ref_end + 1):
        if time.time() > deadline - 5:
            break
        window = sp[n+1:n+25]
        distinct_count = len(set(window.tolist()))
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n

    # If time permits, we can do a broader refinement or a second coarse pass 
    # with a different offset to catch missed peaks.
    if time.time() < deadline - 20:
        # Second coarse pass with offset 50
        for n in range(1000 + 50, N + 1, COARSE_STEP):
            if time.time() > deadline - 5:
                break
            window = sp[n+1:n+25]
            distinct_count = len(set(window.tolist()))
            
            if distinct_count < best_distinct:
                best_distinct = distinct_count
                best_n = n
                
                # Refine immediately around this new best
                ref_start2 = max(1000, best_n - 100)
                ref_end2 = min(N, best_n + 100)
                for n2 in range(ref_start2, ref_end2 + 1):
                    if time.time() > deadline - 2:
                        break
                    window2 = sp[n2+1:n2+25]
                    dc2 = len(set(window2.tolist()))
                    if dc2 < best_distinct:
                        best_distinct = dc2
                        best_n = n2

    return best_n

# EVOLVE-BLOCK-END
