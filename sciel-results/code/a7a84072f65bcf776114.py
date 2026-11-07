"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Determine bounds
    upper_bound = max(pair[1] for pair in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF) to enable quick prime detection and factor checks
    lpf = [0] * (upper_bound + 1)
    for i in range(2, upper_bound + 1):
        if lpf[i] == 0:
            lpf[i] = i
            if i * i <= upper_bound:
                for j in range(i * i, upper_bound + 1, i):
                    if lpf[j] == 0:
                        lpf[j] = i
    
    # Identify all primes up to the maximum gap size for the marking strategy
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = [i for i in range(2, max_gap + 1) if lpf[i] == i]
    
    certified_witnesses = []
    
    for lo, hi in P['gaps']:
        gap_width = hi - lo
        # Find the first integer in (lo, hi) that has no prime factor < gap_width
        # We use a boolean mask to track validity of candidates in this specific interval
        interval_length = hi - lo - 1
        candidates = [True] * interval_length
        
        # Mark candidates that are divisible by any prime smaller than the gap width
        for p in small_primes:
            if p >= gap_width:
                break
            # Find the first multiple of p in (lo, hi)
            # start = smallest multiple of p strictly greater than lo
            start_idx = (lo + 1 + p - 1) // p * p - lo - 1
            if start_idx < 0:
                start_idx = 0
                
            for idx in range(start_idx, interval_length, p):
                candidates[idx] = False
        
        # Locate the first unmarked (valid) candidate
        witness_val = None
        for k, is_valid in enumerate(candidates):
            if is_valid:
                witness_val = lo + 1 + k
                break
        
        certified_witnesses.append(witness_val)
        
    return certified_witnesses
# EVOLVE-BLOCK-END
