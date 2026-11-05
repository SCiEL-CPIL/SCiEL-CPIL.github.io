import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # 1. Generate primes up to 10^6 for fast phi computation
    limit_p = 10**6
    # Sieve of Eratosthenes
    is_prime = np.ones(limit_p + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int64)
    
    # 2. Segmented Sieve for Totients
    # We process a segment of length W
    W = 200000  # Segment size, large enough to contain a run of 200 easily
    
    def compute_phi_segment(L, W):
        """
        Compute phi(n) for n in [L, L+W-1]
        Returns a numpy array of length W
        """
        # phi[n] starts as n
        phi = np.arange(L, L + W, dtype=np.int64)
        # rem[n] stores the remaining part of n after dividing by small primes
        rem = np.arange(L, L + W, dtype=np.int64)
        
        # Process primes up to sqrt(L+W)
        max_p = int(math.isqrt(L + W - 1))
        
        # Filter primes <= max_p
        # primes is sorted
        # Find index where primes > max_p
        idx = np.searchsorted(primes, max_p)
        small_primes = primes[:idx]
        
        for p in small_primes:
            p = int(p)
            # Find first multiple of p in [L, L+W)
            # start = ceil(L/p) * p
            start = ((L + p - 1) // p) * p
            
            # Iterate through multiples
            # indices in the segment: start-L, start-L+p, ...
            for j in range(start - L, W, p):
                if rem[j] % p == 0:
                    # Divide out all factors of p
                    while rem[j] % p == 0:
                        rem[j] //= p
                    # Update phi
                    phi[j] = phi[j] // p * (p - 1)
        
        # After processing small primes, rem[j] is either 1 or a prime factor > sqrt(L+W)
        # If rem[j] > 1, it is a prime factor
        mask = rem > 1
        if np.any(mask):
            # phi[j] = phi[j] // rem[j] * (rem[j] - 1)
            # Vectorized update
            phi[mask] = phi[mask] // rem[mask] * (rem[mask] - 1)
            
        return phi

    # 3. Search for long runs
    best_start = 1
    best_len = 1
    
    # We want to find a run of length 200.
    # Randomly sample segments in the range [10^9, 10^12]
    # s + k - 1 <= 10^12 => s <= 10^12 - 199
    
    max_samples = 1000
    attempts = 0
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < DEADLINE and attempts < max_samples:
        # Pick a random start L
        # L should be large enough to make collisions rare
        # L in [10^9, 10^12 - W]
        L = int(rng.integers(10**9, 10**12 - W))
        
        phi_vals = compute_phi_segment(L, W)
        
        # Scan for longest run of distinct values
        # We are looking for consecutive integers with distinct phi values
        # Note: phi values are for consecutive integers n, n+1, ...
        
        seen = set()
        current_len = 0
        start_idx = 0
        
        for i in range(W):
            v = int(phi_vals[i])
            if v in seen:
                # Run broken
                if current_len > best_len:
                    best_len = current_len
                    best_start = L + start_idx
                # Reset
                seen = set()
                current_len = 0
                start_idx = i
            else:
                seen.add(v)
                current_len += 1
                if current_len > best_len:
                    best_len = current_len
                    best_start = L + start_idx
                if best_len == 200:
                    break
        # Check the last run
        if current_len > best_len:
            best_len = current_len
            best_start = L + start_idx
        
        if best_len == 200:
            break
            
        attempts += 1
        
        # Time check
        if attempts % 10 == 0:
            if time.time() - start_time > DEADLINE:
                break

    # Ensure constraints are met
    if best_start + best_len - 1 > 10**12:
        best_len = 10**12 - best_start + 1
        if best_len < 1:
            best_len = 1
            best_start = 1
            
    if best_start < 1:
        best_start = 1
        
    return {'start': best_start, 'length': best_len}
