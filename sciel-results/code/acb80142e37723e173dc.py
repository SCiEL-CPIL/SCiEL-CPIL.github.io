import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}
    
    # Precompute least prime factor (LPF) for all numbers up to the maximum prime
    hi_max = max(hi for lo, hi in P['gaps'])
    
    # lpf[i] will store the least prime factor of i. 0 means not yet set (or prime itself later).
    lpf = [0] * (hi_max + 1)
    
    # Sieve for LPF
    for i in range(2, hi_max + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            # Mark multiples
            # Start from i*i because smaller multiples have smaller prime factors already marked
            for j in range(i * i, hi_max + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # We need a number m in (lo, hi) such that lpf[m] >= gap
        # Note: If m is prime, lpf[m] = m. Since m > lo and m < hi, m is at least lo+1.
        # The condition is lpf[m] >= gap.
        
        found = False
        # Iterate through candidates in the gap
        for m in range(lo + 1, hi):
            # lpf[m] is the least prime factor.
            # If m is prime, lpf[m] = m.
            # If m is composite, lpf[m] is its smallest prime divisor.
            if lpf[m] >= gap:
                result[i] = int(m)  # Ensure Python int
                found = True
                break
        
        # If not found, result[i] remains None
    
    return result
