import math

def construct():
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Define window size
    W = 200
    max_limit = 10**12
    
    # 3. Calculate a primorial to use as a modulus
    # Using a primorial ensures that s is divisible by many small primes,
    # making phi(s) relatively small, while neighbors are "rough" and have larger phi values.
    # This reduces collision probability.
    primorial = 1
    for p in primes:
        if primorial * p > 10**9: # Keep primorial manageable, ~10^9 is fine
            break
        primorial *= p
        
    # 4. Find starting point s near 10^12 that is a multiple of primorial
    # s = k * primorial
    # We want s + W - 1 <= 10^12
    # So s <= 10^12 - W + 1
    max_s = max_limit - W + 1
    
    # Find the largest multiple of primorial <= max_s
    s = (max_s // primorial) * primorial
    
    # 5. Try a few candidates if the first one fails
    for attempt in range(50):
        start = s - attempt * primorial
        
        if start < 1:
            break
            
        # Segmented sieve for phi in [start, start + W - 1]
        nums = [start + i for i in range(W)]
        phis = [start + i for i in range(W)]
        
        # Sieve with primes
        for p in primes:
            if p * p > start + W - 1:
                break
                
            # Find first multiple of p in [start, start + W - 1]
            m = ((start + p - 1) // p) * p
            
            if m > start + W - 1:
                continue
                
            for x in range(m, start + W, p):
                idx = x - start
                if nums[idx] % p == 0:
                    phis[idx] -= phis[idx] // p
                    while nums[idx] % p == 0:
                        nums[idx] //= p
                        
        # Handle remaining prime factors
        for i in range(W):
            if nums[i] > 1:
                phis[i] -= phis[i] // nums[i]
                
        # Check distinctness
        if len(set(phis)) == W:
            return {'start': start, 'length': W}
            
    # Fallback: If primorial strategy fails, try random-ish large starts
    # Or just return 1
    return {'start': 1, 'length': 1}
