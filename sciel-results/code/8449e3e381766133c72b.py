"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    """
    For each consecutive prime pair (lo, hi), find an integer m strictly between them
    such that the least prime factor of m is at least (hi - lo).
    
    Strategy:
    1. Precompute small primes up to a reasonable bound.
    2. For each gap, the "forbidden" small primes are those < gap_size.
    3. An integer m in (lo, hi) is valid if it is NOT divisible by any prime < gap_size.
       Note: if m is prime, its least prime factor is m itself, which is > lo >= 2, and 
       typically much larger than the gap size. But we need LPF(m) >= gap_size.
       Actually, if m is composite, its least prime factor must be >= gap_size.
       If m is prime, LPF(m) = m. Since m > lo and lo is around 10000, and gap_size is at most ~20,
       m >= gap_size is always true. So primes in the gap are always valid witnesses!
       
    So the simplest valid witness for any gap is just the first prime in (lo, hi).
    But wait, the gap is between consecutive primes, so there are NO primes in (lo, hi)!
    
    Therefore, m must be composite, and its least prime factor must be >= gap_size.
    This means m must not be divisible by any prime < gap_size.
    
    So for each gap (lo, hi) with size g = hi - lo:
    - Let forbidden_primes = all primes < g.
    - Find the first m in (lo, hi) such that m is not divisible by any p in forbidden_primes.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Precompute primes up to a reasonable limit
    # Max gap in our data is 20 (10273 to 10289). So we need primes < 20.
    # But to be safe and general, let's compute primes up to 1000.
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each gap, find a witness
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
            
        gap_size = hi - lo
        # Forbidden primes: all primes < gap_size
        forbidden = [p for p in primes if p < gap_size]
        
        found = False
        # Search from lo+1 to hi-1
        # We can try to be smart: start from lo+1
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
                
            # Check if m is divisible by any forbidden prime
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            
            if valid:
                # Verify: least prime factor of m must be >= gap_size
                # Since m is not divisible by any prime < gap_size, 
                # its least prime factor is either m itself (if prime) or some prime >= gap_size.
                # In either case, LPF(m) >= gap_size.
                # But we must ensure m is an integer (it is) and strictly between lo and hi (it is).
                result[i] = m
                found = True
                break
        
        if not found:
            # No witness found in this gap within time
            result[i] = None
    
    # Ensure types are correct
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            # Ensure it's a Python int, not bool or numpy int
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
