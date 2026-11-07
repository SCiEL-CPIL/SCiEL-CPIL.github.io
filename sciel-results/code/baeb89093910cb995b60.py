"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute primes up to a sufficient limit.
    # Max gap in P['gaps'] is 28. We need primes < 28.
    # But to be safe for any future evolution or if I misread, let's go to 100.
    # Actually, the condition is lpf(m) >= gap.
    # So we need to check if m is divisible by any prime p < gap.
    # Max gap is 28. Primes < 28 are 2, 3, 5, 7, 11, 13, 17, 19, 23.
    primes = _sieve_primes(100)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # We need lpf(m) >= gap.
        # This means m is not divisible by any prime p < gap.
        
        # If gap <= 2, there are no integers strictly between lo and hi (if gap=2, lo+1 is not strictly between? 
        # Wait, strictly between lo and hi means lo < m < hi.
        # If gap = 2, hi = lo + 2. Integers: lo+1.
        # Condition: lpf(lo+1) >= 2. This is always true for any integer > 1.
        # So for gap 2, lo+1 is always a witness?
        # Let's check: "least prime factor is at least the prime gap".
        # Gap = 2. lpf(m) >= 2. Since m is an integer > 1, its lpf is at least 2.
        # So yes, any integer in a gap of 2 is a witness.
        
        # For gap 3: Integers lo+1, lo+2.
        # Need lpf(m) >= 3. So m not divisible by 2.
        
        # General case:
        # Find primes p < gap.
        small_primes = [p for p in primes if p < gap]
        
        # Search for a witness.
        # To maximize chances, we can just iterate.
        # Since the search space is tiny, we can just find the first one.
        # However, different witnesses might be preferred? No, any witness counts as 1.
        # So finding *any* witness is sufficient.
        
        found = False
        # We can scan from the middle outwards, or just linearly.
        # Linear is fine.
        for m in range(lo + 1, hi):
            valid = True
            for p in small_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
        
        # If no witness found, result[i] remains None.
        
    return result
# EVOLVE-BLOCK-END
