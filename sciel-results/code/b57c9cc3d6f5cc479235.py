import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Standard sieve for primes up to limit."""
        if limit < 2:
            return np.array([], dtype=np.int64)
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = False
        return np.where(is_prime)[0]

    def compute_phi_segment(L, B, primes):
        """
        Compute phi(n) for n in [L, L+B-1].
        L: start of segment
        B: size of segment
        primes: array of primes up to sqrt(L+B)
        """
        # Initialize phi and rem arrays
        # n = L + i
        n_vals = L + np.arange(B, dtype=np.int64)
        phi = n_vals.copy()
        rem = n_vals.copy()
        
        # Process each prime
        for p in primes:
            p = int(p)
            # Find first multiple of p in [L, L+B)
            # first = ceil(L / p) * p
            first = (L + p - 1) // p * p
            # Indices in the segment: (first - L) to (L+B-1) with step p
            start_idx = first - L
            if start_idx < 0:
                start_idx = 0
            
            # Create indices array
            # We need to update phi and rem for these indices
            if start_idx >= B:
                continue
                
            indices = np.arange(start_idx, B, p)
            
            # Update phi: phi[i] *= (1 - 1/p) => phi[i] -= phi[i] // p
            # But we must use the CURRENT phi value, which has been updated by previous primes.
            # This is correct for the totient function.
            phi[indices] -= phi[indices] // p
            
            # Update rem: remove all factors of p
            # rem[indices] //= p
            # while rem[indices] % p == 0: rem[indices] //= p
            # Vectorized loop is tricky. 
            # Since p is small, the number of factors is small.
            # We can do:
            # while True:
            #   mask = (rem[indices] % p == 0)
            #   if not np.any(mask): break
            #   rem[indices][mask] //= p
            #   # Note: rem[indices][mask] creates a copy if not careful? 
            #   # rem[indices] is a view? No, fancy indexing returns a copy.
            #   # We need to update the original array.
            
            # Better approach for rem:
            # We can just divide by p once, then check again.
            # Actually, for the final step, we only need to know if rem[i] > 1.
            # If rem[i] > 1, it is a prime factor (since all smaller primes were removed).
            # So we just need to remove ALL factors of p from rem[i].
            
            # Let's do the division in a loop for each index? No, too slow.
            # We can do:
            # current_rem = rem[indices]
            # while True:
            #    divisible = (current_rem % p == 0)
            #    if not np.any(divisible):
            #        break
            #    current_rem[divisible] //= p
            # rem[indices] = current_rem
            
            # This is efficient enough because the number of iterations is small (max 20 for p=2).
            
            current_rem = rem[indices]
            while True:
                divisible = (current_rem % p == 0)
                if not np.any(divisible):
                    break
                current_rem[divisible] //= p
            rem[indices] = current_rem

        # Final step: if rem[i] > 1, it's a prime factor
        # phi[i] -= phi[i] // rem[i]
        mask = rem > 1
        if np.any(mask):
            # We need to update phi for these indices
            # phi[mask] -= phi[mask] // rem[mask]
            # But rem[mask] contains the prime factor.
            # This is correct.
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_distinct_run(phi_vals):
        """
        Find the longest contiguous subsequence of phi_vals with all distinct values.
        Returns (max_length, start_index) where start_index is the 0-based index in phi_vals.
        """
        n = len(phi_vals)
        if n < 1:
            return 0, 0
            
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = phi_vals[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Primes up to sqrt(10^12 + 10^7) ~ 10^6
    # 10^6 is safe.
    prime_limit = 1_000_000
    primes = sieve_primes(prime_limit)
    
    # Strategy: Search near 10^12
    # Block size 10^7
    B = 10_000_000
    
    # Try a few segments
    # 1. Near 10^12
    # 2. Near 10^11
    # 3. Near 10^9
    
    segments = [
        10**12 - B,
        10**11 - B,
        10**9 - B
    ]
    
    # Ensure L >= 1
    segments = [max(1, s) for s in segments]
    
    for L in segments:
        if time_remaining() < 10.0:
            break
            
        try:
            # Check if L+B-1 <= 10^12
            if L + B - 1 > 10**12:
                # Adjust L to be 10^12 - B + 1
                L = 10**12 - B + 1
                if L < 1:
                    L = 1
                    B = 10**12 # This won't happen, 10^12 > 10^7
            
            phi_vals = compute_phi_segment(L, B, primes)
            
            # Find longest run
            k, idx = find_longest_distinct_run(phi_vals)
            
            # Convert idx to actual start s
            s = L + idx
            
            # Cap k at 200
            effective_k = min(k, 200)
            
            # Verify constraints
            if s >= 1 and s + effective_k - 1 <= 10**12:
                if effective_k > best_k:
                    best_k = effective_k
                    best_s = s
                    
        except MemoryError:
            break
        except Exception:
            break

    # Fallback if no segment was processed or failed
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    return {'start': best_s, 'length': best_k}
