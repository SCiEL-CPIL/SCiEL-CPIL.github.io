"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def phi(x):
        """Compute Euler's totient function for a single number."""
        if x <= 1:
            return 0
        result = x
        n = x
        d = 2
        while d * d <= n:
            if n % d == 0:
                result -= result // d
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            result -= result // n
        return result

    def phi_segment(start, end):
        """Compute phi values for range [start, end] using segmented approach."""
        n = end - start + 1
        if n <= 0:
            return []
        # For small ranges, just compute individually
        if n <= 10000:
            return [phi(x) for x in range(start, end + 1)]
        
        # For larger ranges, use a segmented sieve approach
        # We need primes up to sqrt(end)
        limit = int(math.isqrt(end)) + 1
        # Sieve primes up to limit
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, limit + 1) if is_prime[i]]
        
        # Initialize phi values
        phi_vals = list(range(start, end + 1))
        remaining = list(range(start, end + 1))
        
        for p in primes:
            # Find multiples of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            for m in range(first_mult, end + 1, p):
                idx = m - start
                if phi_vals[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                # Mark that p has been processed for this number
                # We need to track which primes have been used
                # Actually, let's use a different approach
        
        # Simpler: for each number, divide out prime factors
        # But that's O(n * sqrt(n)) which is too slow
        # Let's use the individual phi for now if segment is not too large
        if n <= 100000:
            return [phi(x) for x in range(start, end + 1)]
        
        # For very large segments, we need a better method
        # Use the fact that phi(n) = n * prod(1 - 1/p) for prime p|n
        # We can compute this by tracking prime factors
        phi_vals = list(range(start, end + 1))
        # For each prime, divide out its contribution
        for p in primes:
            first_mult = ((start + p - 1) // p) * p
            for m in range(first_mult, end + 1, p):
                idx = m - start
                if phi_vals[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining prime factors > sqrt(end)
        # For numbers that still have a factor > 1 after dividing by all primes <= sqrt(end)
        # that factor is prime
        for idx in range(n):
            val = phi_vals[idx]
            orig = start + idx
            # Check if orig has a prime factor > sqrt(end)
            # Actually, we already divided by all primes <= sqrt(end)
            # So if phi_vals[idx] != orig * prod(1-1/p) for p|orig, p<=sqrt(end)
            # We need to check if there's a remaining prime factor
            # This is tricky. Let's just use individual phi for safety
            pass
        
        # For safety, use individual computation for now
        # This is slower but correct
        return [phi(x) for x in range(start, end + 1)]

    # Strategy: search for longest run of consecutive integers with distinct phi values
    # Start with a small range and expand
    best_k = 1
    best_s = 1
    
    # First, search in a moderate range to find a good starting point
    # Then try to extend from there
    
    # Search in chunks
    chunk_size = 50000
    max_search = 10**9  # Search up to 10^9 initially
    
    s = 1
    while s <= max_search and time_left() > 5.0:
        end = min(s + chunk_size - 1, max_search)
        if end < s:
            break
        
        # Compute phi values for this chunk
        vals = phi_segment(s, end)
        
        # Find longest run of distinct phi values in this chunk
        seen = set()
        current_k = 0
        current_start = s
        
        for i, v in enumerate(vals):
            if v in seen:
                # Collision - end current run
                if current_k > best_k:
                    best_k = current_k
                    best_s = current_start
                seen.clear()
                current_k = 0
                current_start = s + i
            seen.add(v)
            current_k += 1
        
        # Check the last run
        if current_k > best_k:
            best_k = current_k
            best_s = current_start
        
        s = end + 1
    
    # Now try to extend from the best found
    # Try to extend both forward and backward from best_s
    if best_k > 1 and time_left() > 10.0:
        # Extend forward
        forward_start = best_s + best_k
        if forward_start <= 10**12:
            seen = set(phi(x) for x in range(best_s, best_s + best_k))
            ext_k = best_k
            ext_s = best_s
            x = forward_start
            while x <= 10**12 and time_left() > 5.0:
                v = phi(x)
                if v in seen:
                    break
                seen.add(v)
                ext_k += 1
                ext_s = best_s
                x += 1
                if ext_k >= 200:
                    break
            if ext_k > best_k:
                best_k = ext_k
                best_s = ext_s
        
        # Extend backward
        if best_s > 1:
            backward_end = best_s - 1
            seen = set(phi(x) for x in range(best_s, best_s + best_k))
            ext_k = best_k
            ext_s = best_s
            x = backward_end
            while x >= 1 and time_left() > 5.0:
                v = phi(x)
                if v in seen:
                    break
                seen.add(v)
                ext_k += 1
                ext_s = x
                x -= 1
                if ext_k >= 200:
                    break
            if ext_k > best_k:
                best_k = ext_k
                best_s = ext_s
    
    # Ensure constraints are met
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
        best_s = max(1, best_s + (200 - best_k))
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
