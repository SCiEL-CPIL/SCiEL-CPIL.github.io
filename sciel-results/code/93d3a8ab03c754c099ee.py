"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def sieve_primes(limit):
        """Sieve of Eratosthenes to find all primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segmented(lo, hi, primes):
        """
        Compute phi(n) for all n in [lo, hi] using segmented approach.
        Returns a list of phi values.
        """
        n = hi - lo + 1
        if n <= 0:
            return []
        
        # Initialize phi and remaining arrays
        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))
        
        # Handle phi(1) = 1 if lo == 1
        if lo == 1:
            phi[0] = 1
            remaining[0] = 1
        
        # For each prime p, update phi and remaining for multiples of p
        for p in primes:
            if p > hi:
                break
            # Find the first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > sqrt(hi)
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def find_best_run(phi_arr, offset, max_len=200):
        """
        Find the longest run of distinct phi values in phi_arr.
        Returns (start_index_in_original, length).
        """
        n = len(phi_arr)
        best_k = 0
        best_s = offset
        
        # Use a sliding window approach with a set
        # For each starting position, extend as far as possible
        # But this is O(n * max_len) which is fine for reasonable n
        
        for i in range(n):
            if n - i < best_k:
                break  # Can't beat best_k with remaining elements
            seen = set()
            k = 0
            while i + k < n and k < max_len and phi_arr[i + k] not in seen:
                seen.add(phi_arr[i + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = offset + i
        
        return best_s, best_k
    
    # Precompute primes up to sqrt(10^12) = 10^6
    # We might not need all of them, but let's get a good set
    # Actually, for segmented phi, we need primes up to sqrt(hi)
    # For hi up to 10^12, we need primes up to 10^6
    
    # Let's start with a moderate range to get a baseline
    best_s, best_k = 1, 1
    
    # Strategy: Try different ranges
    # 1. Small range [1, 100000] for quick baseline
    # 2. Larger ranges at different scales
    # 3. Random ranges in the upper part of [1, 10^12]
    
    # First, get primes up to 10^6 (enough for sqrt(10^12))
    # But we'll compute primes up to sqrt(hi) for each segment
    
    # Try a few fixed ranges first
    test_ranges = [
        (1, 100000),
        (1, 500000),
        (1, 1000000),
        (1000000, 2000000),
        (10000000, 11000000),
        (100000000, 101000000),
    ]
    
    for lo, hi in test_ranges:
        if time_left() < 5.0:
            break
        try:
            # Get primes up to sqrt(hi)
            prime_limit = int(math.isqrt(hi)) + 1
            primes = sieve_primes(prime_limit)
            phi_arr = compute_phi_segmented(lo, hi, primes)
            if time_left() < 3.0:
                break
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Now try random ranges in the upper part of [1, 10^12]
    # Longer runs are more likely for larger numbers
    random.seed(42)
    max_n = 10**12
    
    # Try random segments of size ~10000-100000
    for _ in range(50):  # Try up to 50 random segments
        if time_left() < 10.0:
            break
        # Choose a random start in [1, max_n - 100000]
        seg_size = random.randint(10000, 100000)
        if max_n - seg_size < 1:
            break
        lo = random.randint(1, max_n - seg_size)
        hi = lo + seg_size - 1
        
        try:
            prime_limit = int(math.isqrt(hi)) + 1
            # Only sieve up to a reasonable limit; if prime_limit is too large, skip
            if prime_limit > 2000000:
                # Use a smaller prime set; numbers with large prime factors
                # will be handled by the "remaining" logic
                prime_limit = 2000000
            primes = sieve_primes(prime_limit)
            phi_arr = compute_phi_segmented(lo, hi, primes)
            if time_left() < 5.0:
                break
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
