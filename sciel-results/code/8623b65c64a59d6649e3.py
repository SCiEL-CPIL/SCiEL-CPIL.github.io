"""Rewritten program with optimized witness search."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin within 60s limit
    
    # Precompute primes up to sqrt(max_value) for efficiency
    # Max value is 10289, sqrt is ~101
    max_val = max(hi for _, hi in P['gaps'])
    limit = int(math.isqrt(max_val)) + 1
    
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # List of all small primes up to limit
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    result = []
    
    for lo, hi in P['gaps']:
        if time.time() > deadline:
            # If we hit the deadline, append None for remaining to ensure valid structure
            # though this shouldn't happen for this input size
            result.append(None)
            continue
            
        gap = hi - lo
        
        # Identify primes strictly less than the gap
        # If m is not divisible by any of these, its LPF is >= gap
        # (Note: If m is prime, LPF is m, which is > lo >= 10007 > gap)
        # So we only need to check divisibility by primes < gap
        relevant_primes = [p for p in small_primes if p < gap]
        
        witness = None
        for m in range(lo + 1, hi):
            # Check if m is divisible by any relevant prime
            is_valid = True
            for p in relevant_primes:
                if m % p == 0:
                    is_valid = False
                    break
            
            if is_valid:
                witness = m
                break
        
        result.append(witness)
        
    # Ensure types are correct
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
            
    return final_result
# EVOLVE-BLOCK-END
