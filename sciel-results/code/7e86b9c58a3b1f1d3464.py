"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def construct():
    import math
    
    def sieve_primes(limit):
        """Generate all primes up to limit using sieve of Eratosthenes."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    # Precompute primes up to 10^6 (sufficient for segmented sieve of numbers up to ~10^12)
    PRIME_LIMIT = 1000000
    primes = sieve_primes(PRIME_LIMIT)
    
    best_k = 1
    best_s = 1
    deadline = _START_TIME + _TIME_LIMIT - 5.0  # 5s safety margin
    
    MAX_K = 200
    
    def segmented_phi(start, end):
        """
        Compute phi(n) for all n in [start, end] using segmented sieve.
        Returns list of phi values.
        """
        n = end - start + 1
        if n <= 0:
            return []
        
        # Initialize: rem[i] = start + i, phi[i] = start + i
        rem = list(range(start, end + 1))
        phi = list(range(start, end + 1))
        
        # For each prime, mark multiples in the segment
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            if start % p == 0:
                first = start
            else:
                first = start + (p - start % p) % p
            
            # Mark all multiples of p
            for j in range(first - start, n, p):
                if rem[j] % p == 0:
                    phi[j] = phi[j] // p * (p - 1)
                    while rem[j] % p == 0:
                        rem[j] //= p
        
        # Handle remaining prime factors > sqrt(end)
        for i in range(n):
            if rem[i] > 1:
                phi[i] = phi[i] // rem[i] * (rem[i] - 1)
        
        return phi
    
    def search_in_range(start, end, max_k):
        """Search for the longest run of distinct phi values in [start, end]."""
        n = end - start + 1
        if n <= 0:
            return 1, start
        
        # Check time before computing
        if time.time() > deadline:
            return 1, start
        
        # Compute phi values using segmented sieve
        phi_vals = segmented_phi(start, end)
        
        if time.time() > deadline:
            return 1, start
        
        # Use sliding window to find longest run of distinct values
        best_local_k = 1
        best_local_s = start
        
        seen = {}  # value -> index
        left = 0
        
        for right in range(n):
            if time.time() > deadline:
                break
            
            val = phi_vals[right]
            
            # If val is already in the window, move left past its previous occurrence
            if val in seen:
                left = max(left, seen[val] + 1)
            
            seen[val] = right
            
            # Current window [left, right] has all distinct values
            current_k = right - left + 1
            
            if current_k > best_local_k:
                best_local_k = current_k
                best_local_s = start + left
            
            # If we've hit max_k, we can stop early
            if best_local_k >= max_k:
                break
        
        return best_local_k, best_local_s
    
    # Search strategy: use larger contiguous blocks with segmented sieve
    # Segmented sieve is much faster than per-number trial division
    
    ranges_to_try = [
        # Start with medium-sized blocks
        (1, 200000),
        (1000000, 1200000),
        (10000000, 10200000),
        (100000000, 100200000),
        (1000000000, 1000200000),
        (10000000000, 10000200000),
        (100000000000, 100000200000),
        (999999000000, 10**12),
    ]
    
    for start, end in ranges_to_try:
        if time.time() > deadline:
            break
        if end > 10**12:
            end = 10**12
        if start < 1:
            start = 1
        if start > end:
            continue
        
        k_found, s_found = search_in_range(start, end, MAX_K)
        if k_found > best_k:
            best_k = k_found
            best_s = s_found
        
        # If we found a reasonably long run, try to extend it
        if k_found >= 5:
            # Try extending left and right with larger margin
            left_start = max(1, s_found - 2000)
            right_end = min(10**12, s_found + k_found + 2000)
            extended_k, extended_s = search_in_range(left_start, right_end, MAX_K)
            if extended_k > best_k:
                best_k = extended_k
                best_s = extended_s
    
    # If time remains, do additional targeted searches with larger blocks
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        # Try several random positions with larger block sizes
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            # Pick a random start in a large range
            base = random.randint(10**6, 10**12 - 200000)
            s = base
            e = min(10**12, base + 200000)
            
            k_found, s_found = search_in_range(s, e, MAX_K)
            if k_found > best_k:
                best_k = k_found
                best_s = s_found
            
            # If promising, extend
            if k_found >= 5:
                left_start = max(1, s_found - 2000)
                right_end = min(10**12, s_found + k_found + 2000)
                extended_k, extended_s = search_in_range(left_start, right_end, MAX_K)
                if extended_k > best_k:
                    best_k = extended_k
                    best_s = extended_s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
