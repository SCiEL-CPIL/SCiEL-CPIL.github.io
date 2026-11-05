import numpy as np
import math

def construct():
    # 1. Generate primes up to 10^6 using NumPy for speed
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Define window size
    W = 200
    max_limit = 10**12
    
    # 3. Search strategy
    # Start near the upper bound.
    # Try a few consecutive starts to ensure we find a valid window.
    for attempt in range(20):
        s = max_limit - W + 1 - attempt
        
        # 4. Segmented Sieve for Phi using NumPy
        # Use int64 to hold numbers up to 10^12
        nums = np.arange(s, s + W, dtype=np.int64)
        phis = nums.copy()
        rem = nums.copy()
        
        # Process primes
        # We only need primes up to sqrt(s + W - 1)
        # sqrt(10^12) = 10^6
        # All primes in our list are <= 10^6, so we check all of them.
        # But we can break early if p > s + W - 1 (not needed here as s is large)
        
        for p in primes:
            if p > s + W - 1:
                break
                
            # Find indices where rem is divisible by p
            mask = (rem % p == 0)
            if not np.any(mask):
                continue
            
            # Update phi: phi(n) = phi(n) * (1 - 1/p) = phi(n) - phi(n)//p
            phis[mask] -= phis[mask] // p
            
            # Remove all factors of p from rem
            # Get indices of elements divisible by p
            idxs = np.where(mask)[0]
            for idx in idxs:
                while rem[idx] % p == 0:
                    rem[idx] //= p
                    
        # Handle remaining prime factors
        # If rem[i] > 1, it is a prime factor q.
        # phi(n) -= phi(n) // q
        mask_rem = (rem > 1)
        if np.any(mask_rem):
            phis[mask_rem] -= phis[mask_rem] // rem[mask_rem]
            
        # Check distinctness
        if len(np.unique(phis)) == W:
            return {'start': int(s), 'length': W}
            
    # Fallback: If we failed to find a run of 200 (extremely unlikely),
    # try smaller lengths or return a minimal valid solution.
    # For robustness, we can try to find a sub-window of the last computed phis?
    # But we didn't store them. Let's just return a small valid run.
    # A run of length 1 is always valid.
    # But we want to maximize k.
    # We can try a few more random starts or just return 1.
    # Given the high probability of success for k=200, this fallback is rarely hit.
    
    return {'start': 1, 'length': 1}
