import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Precompute least prime factor (LPF) up to the maximum prime in the gaps
    max_prime = max(hi for _, hi in P['gaps'])
    lpf = list(range(max_prime + 1))
    for i in range(2, int(max_prime**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_prime + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    results = []
    
    for lo, hi in P['gaps']:
        gap = hi - lo
        witness = None
        
        # Check each integer strictly between lo and hi
        for m in range(lo + 1, hi):
            # lpf[m] is the least prime factor of m
            # We need lpf(m) >= gap
            # For m=1, lpf[1]=1; for primes, lpf[m]=m
            # If m is prime, lpf[m] = m which is > gap (since m < hi and hi-lo=gap, m can be up to hi-1)
            # Actually m is strictly between lo and hi, so m < hi = lo + gap
            # If m is prime, lpf[m] = m. We need m >= gap.
            # If m is composite, lpf[m] is its smallest prime factor.
            
            if lpf[m] >= gap:
                witness = m
                break
        
        if witness is not None:
            results.append(int(witness))
        else:
            results.append(None)
    
    return results
