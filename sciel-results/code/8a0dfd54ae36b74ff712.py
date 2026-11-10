"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def construct():
    def time_left():
        return _TIME_LIMIT - (time.time() - _START_TIME)
    
    def sieve_primes(limit):
        """Generate all primes up to limit using sieve of Eratosthenes."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_sieve(n):
        """Compute phi(1) through phi(n) using a sieve-based approach."""
        phi = list(range(n + 1))
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    def compute_phi_segmented(start, end, primes):
        """Compute phi(x) for x in [start, end] using precomputed primes."""
        n = end - start + 1
        # For each number in range, we'll factor using primes up to sqrt(end)
        phi_vals = list(range(start, end + 1))
        remaining = list(range(start, end + 1))
        
        sqrt_end = int(math.isqrt(end))
        for p in primes:
            if p > sqrt_end:
                break
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            for m in range(first_mult, end + 1, p):
                idx = m - start
                if remaining[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining primes > sqrt(end)
        for i in range(n):
            if remaining[i] > 1:
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
        return phi_vals
    
    def find_best_run(phi_vals, start_offset, max_length=200):
        """Find the longest run of consecutive distinct phi values."""
        n = len(phi_vals)
        best_len = 1
        best_start = start_offset
        
        for s in range(n):
            seen = set()
            k = 0
            while s + k < n and s + k + start_offset - start_offset < n:
                val = phi_vals[s + k]
                if val in seen:
                    break
                seen.add(val)
                k += 1
                if k >= max_length:
                    break
            if k > best_len:
                best_len = k
                best_start = start_offset + s
        
        return best_len, best_start
    
    # Strategy:
    # 1. First, compute phi for a moderate range to get a good initial answer
    # 2. Then, search in larger ranges using segmented phi computation
    # 3. Use time budget to explore as much as possible
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Small range to get a quick answer
    n1 = 100000
    phi1 = compute_phi_sieve(n1)
    k1, s1 = find_best_run(phi1, 1, 200)
    if k1 > best_k:
        best_k = k1
        best_s = s1
    
    if time_left() < 5:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Search in larger ranges
    # We need primes up to sqrt(10^12) = 10^6
    max_prime_needed = int(math.isqrt(10**12)) + 1
    primes = sieve_primes(max_prime_needed)
    
    # Search in chunks of increasing size
    chunk_sizes = [100000, 200000, 500000, 1000000, 2000000, 5000000]
    
    for chunk_size in chunk_sizes:
        if time_left() < 10:
            break
        
        # Search in multiple locations across the range [1, 10^12]
        # Focus on ranges where long runs are likely to occur
        # Based on theory, long runs of distinct totients tend to occur
        # in ranges with many small prime factors
        
        # Try several starting points
        start_points = [
            10**6, 10**7, 10**8, 10**9, 10**10,
            10**6 + chunk_size//2, 10**8 + chunk_size//2,
            5*10**6, 5*10**7, 5*10**8, 5*10**9,
            2*10**6, 2*10**7, 2*10**8, 2*10**9,
            3*10**6, 3*10**7, 3*10**8, 3*10**9,
            7*10**6, 7*10**7, 7*10**8, 7*10**9,
            11*10**6, 11*10**7, 11*10**8,
            13*10**6, 13*10**7, 13*10**8,
            17*10**6, 17*10**7, 17*10**8,
            19*10**6, 19*10**7, 19*10**8,
        ]
        
        for sp in start_points:
            if time_left() < 10:
                break
            if sp + chunk_size > 10**12:
                continue
            
            end = min(sp + chunk_size - 1, 10**12)
            phi_chunk = compute_phi_segmented(sp, end, primes)
            k, s = find_best_run(phi_chunk, sp, 200)
            
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 3: If we have time, do a more systematic search
    # Focus on ranges that are products of many small primes (primorials)
    if time_left() > 20:
        # Generate primorials and search near them
        primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]
        
        for p in primorials:
            if time_left() < 15:
                break
            
            # Search in a window around the primorial
            window_size = min(50000, int(math.isqrt(p)) + 1000)
            start = max(1, p - window_size)
            end = min(10**12, p + window_size)
            
            if start >= end:
                continue
            
            phi_chunk = compute_phi_segmented(start, end, primes)
            k, s = find_best_run(phi_chunk, start, 200)
            
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 4: Random sampling in large ranges
    if time_left() > 15:
        import random
        random.seed(42)
        
        for _ in range(50):
            if time_left() < 10:
                break
            
            # Random start in [1, 10^12]
            start = random.randint(1, 10**12 - 200000)
            end = min(start + 200000, 10**12)
            
            phi_chunk = compute_phi_segmented(start, end, primes)
            k, s = find_best_run(phi_chunk, start, 200)
            
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
