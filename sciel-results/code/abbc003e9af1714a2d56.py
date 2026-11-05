import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s limit
    
    # Precompute primes up to 10^6
    # sqrt(10^12) = 10^6
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    if limit_p >= 1:
        is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = 0
    
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    def segmented_phi(s, k):
        """
        Compute phi(s), phi(s+1), ..., phi(s+k-1) using segmented sieve.
        """
        if k <= 0:
            return []
        
        # remaining[i] holds the part of s+i not yet factored by small primes
        # phi_vals[i] holds the running product for phi(s+i)
        remaining = [s + i for i in range(k)]
        phi_vals = [s + i for i in range(k)]
        
        max_n = s + k - 1
        sqrt_max = int(math.isqrt(max_n)) + 1
        
        for p in primes:
            if p > sqrt_max:
                break
            
            # Find the first multiple of p in [s, s+k-1]
            # m = ceil(s/p) * p
            first_mult = ((s + p - 1) // p) * p
            
            for m in range(first_mult, max_n + 1, p):
                idx = m - s
                if remaining[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(k):
            if remaining[i] > 1:
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
        return phi_vals

    best_k = 1
    best_s = 1
    
    # Phase 1: Random sampling of large numbers
    # We look for runs of length 200.
    # For large n, collisions are rare.
    random.seed(42)
    
    # We will sample from ranges [10^8, 10^12]
    # We check a window of 200.
    # If all distinct, k=200.
    # If collision at index j, k=j.
    
    # To be efficient, we can just check if the first 200 are distinct.
    # If yes, we found k=200.
    
    num_attempts = 0
    while time.time() < DEADLINE - 5:
        num_attempts += 1
        
        # Random start point
        # Bias towards larger numbers where collisions are less likely
        s = random.randint(10**8, 10**12 - 200)
        
        # Compute phi for 200 numbers
        phi_values = segmented_phi(s, 200)
        
        # Check for duplicates
        seen = set()
        k = 0
        for val in phi_values:
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
        
        # Optional: Local optimization if we found a long run but not 200
        # But given the nature of the problem, finding 200 is likely.
        # If we find k=199, we could try s+1 or s-1, but random sampling is faster.
        
    return {'start': best_s, 'length': best_k}
