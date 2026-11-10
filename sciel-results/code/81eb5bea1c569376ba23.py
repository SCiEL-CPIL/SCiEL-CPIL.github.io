import math
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Define window size
    # Larger window increases chance of finding long run, but increases sieve cost.
    # Cost is dominated by number of primes.
    # Let's try W = 200,000.
    W = 200_000
    max_val = 10**12
    
    # Split primes for efficiency
    # Small primes: p <= W. We can use vectorized slicing.
    # Large primes: p > W. At most one multiple in window.
    small_primes = primes[primes <= W]
    large_primes = primes[primes > W]
    
    best_k = 1
    best_s = 1
    
    # Random search
    random.seed(42)
    
    def compute_phi_window(lo):
        """Compute phi for [lo, lo+W)"""
        hi = lo + W - 1
        # Ensure we don't go over 10^12
        if hi > max_val:
            # Adjust W or lo? 
            # Since lo is random, just skip if invalid? 
            # Better: ensure lo is chosen such that lo+W-1 <= max_val
            pass
        
        n = W
        phi = np.arange(lo, lo + n, dtype=np.int64)
        rem = np.arange(lo, lo + n, dtype=np.int64)
        
        # Process small primes
        for p in small_primes:
            p = int(p)
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            
            # Indices in phi/rem arrays
            # start is global index, lo is global start
            # idx_start = start - lo
            # idx_end = hi - lo + 1 (exclusive)
            # Step is p
            
            indices = np.arange(start, hi + 1, p) - lo
            
            phi[indices] -= phi[indices] // p
            
            # Update rem: divide by p as many times as possible
            # This is the expensive part.
            # r = rem[indices].copy()
            # while True:
            #   mask = r % p == 0
            #   if not np.any(mask): break
            #   r[mask] //= p
            # rem[indices] = r
            
            # Optimization: Instead of while loop, we can just divide once?
            # No, rem needs to be fully stripped of p to detect remaining prime factors correctly?
            # Actually, for phi calculation, we just need to know if p divides n.
            # phi[n] = n * (1 - 1/p) * ...
            # The `rem` array is used to handle the case where a number has a prime factor > sqrt(hi).
            # If we don't strip rem completely, we might miss that rem is prime.
            # However, if we only care about phi, we can compute it differently.
            # phi[n] = n
            # for each prime p | n: phi[n] -= phi[n] // p
            # We don't strictly need `rem` if we just iterate all primes p <= sqrt(n).
            # But we stop at sqrt(hi).
            # If n has a factor > sqrt(hi), it is prime (or 1).
            # So after processing all p <= sqrt(hi), if rem > 1, then rem is a prime factor.
            # So we MUST strip rem of small primes to find the remaining large prime factor.
            
            # Vectorized stripping is tricky.
            # For small W, Python loop might be okay?
            # Let's stick to the current method but optimize.
            
            r = rem[indices].copy()
            # Unroll the while loop a bit?
            # Most numbers are not divisible by p^2.
            # Check if any are divisible.
            mask = r % p == 0
            if np.any(mask):
                r[mask] //= p
                # Check again for p^2
                mask = r[mask] % p == 0
                if np.any(mask):
                    # Rare
                    r_idx = np.where(mask)[0]
                    # This is getting complex.
                    # Let's just use the while loop, it's usually 1-2 iterations.
                    pass
            
            # Simpler:
            r = rem[indices].copy()
            while True:
                mask = r % p == 0
                if not np.any(mask):
                    break
                r[mask] //= p
            rem[indices] = r

        # Process large primes
        # For p > W, there is at most one multiple in [lo, hi]
        # m = ceil(lo/p) * p
        # if m <= hi:
        #   idx = m - lo
        #   phi[idx] -= phi[idx] // p
        #   rem[idx] //= p (only once, since p^2 > W >= hi-lo+1, so p^2 cannot divide any number in window? 
        #   Wait, if p > W, then p^2 > W^2. The number n < 10^12.
        #   If p ~ 10^6, p^2 = 10^12.
        #   So p can divide n at most once if p > sqrt(n)?
        #   No, if p <= sqrt(n), it can divide multiple times.
        #   But we only process p > W.
        #   If W = 200,000, sqrt(10^12) = 10^6.
        #   So for p in (200,000, 10^6], p can divide n multiple times?
        #   p^2 = 4 * 10^10 < 10^12. So yes, p^2 can divide n.
        #   So we must strip rem completely for large primes too?
        #   But there is only 1 index to update.
        #   So we can do a small Python loop for that index.
        
        for p in large_primes:
            p = int(p)
            if p * p > hi:
                # If p^2 > hi, then p can divide n at most once?
                # No, if p^2 > hi, then n < p^2, so n cannot be divisible by p^2.
                # So we only divide once.
                # We can break early? No, smaller p might still have multiples.
                # But for p > sqrt(hi), no multiple exists?
                # Wait, if p > sqrt(hi), then p^2 > hi.
                # So p can be a factor, but p^2 cannot.
                # So we just divide once.
                pass
            
            k = (lo + p - 1) // p
            m = k * p
            if m <= hi:
                idx = m - lo
                if rem[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    # Strip rem
                    while rem[idx] % p == 0:
                        rem[idx] //= p

        # Handle remaining prime factors in rem
        mask = rem > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    # Search loop
    while time_left() > 1.0:
        # Pick random start
        # Ensure lo + W - 1 <= max_val
        max_lo = max_val - W + 1
        if max_lo < 1:
            break
        lo = random.randint(1, max_lo)
        
        phi = compute_phi_window(lo)
        
        # Find longest distinct run
        last_seen = {}
        start_idx = 0
        max_run = 0
        max_start = 0
        
        for i in range(len(phi)):
            v = phi[i]
            if v in last_seen:
                if last_seen[v] >= start_idx:
                    start_idx = last_seen[v] + 1
            last_seen[v] = i
            
            current_run = i - start_idx + 1
            if current_run > max_run:
                max_run = current_run
                max_start = start_idx
                
        if max_run > best_k:
            best_k = max_run
            best_s = lo + max_start
            if best_k >= 200:
                return {'start': best_s, 'length': best_k}
                
        # Optional: Local refinement around best_s
        # If we found a good run, check neighbors?
        # The random search might jump away.
        # We can store best_s and occasionally re-check around it?
        # Or just let random search do it.

    return {'start': best_s, 'length': best_k}
