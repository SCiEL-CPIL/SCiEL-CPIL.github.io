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
    
    # 3. Construct a primorial to align starting points
    # Product of primes up to 29 is ~ 6.4e8
    # This ensures s has many small factors, making phi(s) "small"
    # Neighbors s+i (i>0) are not divisible by these small primes,
    # making their phi values "large" and distinct from phi(s).
    primorial_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    P = 1
    for p in primorial_primes:
        P *= p
        
    # We will search for a multiple of P near 10^12
    # s = k * P
    # k_max = 10^12 // P
    k_max = max_limit // P
    
    # Try a range of k values downwards from k_max
    # If we try 2000 values, we cover a range of 2000 * P ~ 1.2 * 10^12
    # This covers the entire valid range [1, 10^12] if we start high enough?
    # 10^12 / P is approx 1500.
    # So there are only ~1500 multiples of P in [1, 10^12].
    # We can iterate through ALL of them!
    # 1500 windows * O(W log log W) is very fast.
    
    # Iterate k from k_max down to 1
    # But maybe start near the top as required by "Long runs ... up to 10^12"
    # The task doesn't require s to be large, just s+k-1 <= 10^12.
    # But larger s is generally "safer" for distinctness due to larger value range.
    
    for k in range(k_max, 0, -1):
        start = k * P
        if start + W - 1 > max_limit:
            continue
            
        # Segmented sieve for phi in [start, start+W-1]
        nums = [start + i for i in range(W)]
        phis = [start + i for i in range(W)]
        
        for p in primes:
            if p * p > start + W - 1:
                break
            
            # Find first multiple of p in [start, start+W-1]
            # m = ceil(start / p) * p
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
            
    # Fallback if no multiple of P worked (unlikely)
    # Try random starts or consecutive starts
    # But 1500 checks should be enough.
    # If we reach here, try a few consecutive integers near 10^12
    for offset in range(1000):
        start = max_limit - offset
        if start < 1:
            break
        if start + W - 1 > max_limit:
            continue
            
        nums = [start + i for i in range(W)]
        phis = [start + i for i in range(W)]
        
        for p in primes:
            if p * p > start + W - 1:
                break
            m = ((start + p - 1) // p) * p
            if m > start + W - 1:
                continue
            for x in range(m, start + W, p):
                idx = x - start
                if nums[idx] % p == 0:
                    phis[idx] -= phis[idx] // p
                    while nums[idx] % p == 0:
                        nums[idx] //= p
                        
        for i in range(W):
            if nums[i] > 1:
                phis[i] -= phis[i] // nums[i]
                
        if len(set(phis)) == W:
            return {'start': start, 'length': W}
            
    # Absolute fallback
    return {'start': 1, 'length': 1}
