"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Strategy: Use a sieve to compute phi for a large range, then scan for longest run
    # with distinct values. We'll search in windows to stay within time limits.
    
    # First, let's try a moderate range that we can compute phi for quickly
    # Using a segmented sieve approach for phi
    
    def compute_phi_segment(start, length, primes):
        """Compute phi for range [start, start+length-1] using given primes."""
        phi = list(range(start, start + length))
        for p in primes:
            if p * p > start + length - 1:
                break
            # Find first multiple of p in range
            first = ((start + p - 1) // p) * p
            for j in range(first, start + length, p):
                idx = j - start
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        # Handle remaining prime factors
        for i in range(length):
            if phi[i] > 1:
                phi[i] -= phi[i] // phi[i]
        return phi
    
    # Generate primes up to sqrt(10^12) = 10^6
    # But we only need primes up to sqrt(max_value_in_range)
    # For a range of size N starting at S, we need primes up to sqrt(S+N)
    
    # Let's use a simpler approach: compute phi for a range using trial division
    # with precomputed small primes
    
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    # Precompute primes up to 10^6
    primes = sieve_primes(1000000)
    
    best_k = 1
    best_s = 1
    
    # Try different ranges
    # Start with smaller ranges and increase
    range_sizes = [50000, 100000, 200000, 500000, 1000000]
    start_positions = [1, 100000, 1000000, 10000000, 100000000, 1000000000, 5000000000, 50000000000]
    
    for size in range_sizes:
        if time.time() > deadline - 30:
            break
        for start in start_positions:
            if start + size > 10**12:
                continue
            if time.time() > deadline - 30:
                break
            
            phi_vals = compute_phi_segment(start, size, primes)
            
            # Find longest run of distinct values
            seen = {}
            current_k = 0
            for i in range(size):
                if phi_vals[i] not in seen:
                    seen[phi_vals[i]] = i
                    current_k += 1
                else:
                    if current_k > best_k:
                        best_k = current_k
                        best_s = start + i - current_k
                    # Reset: keep only values from seen[phi_vals[i]]+1 onwards
                    # But this is complex; simpler to restart
                    seen = {}
                    current_k = 0
            
            if current_k > best_k:
                best_k = current_k
                best_s = start + size - current_k
    
    # If we found a good result, verify it
    if best_k > 1:
        # Verify the result
        verify_start = best_s
        verify_k = best_k
        seen = set()
        valid = True
        for i in range(verify_k):
            x = verify_start + i
            # Compute phi(x) directly
            y = x
            result = x
            for p in primes:
                if p * p > y:
                    break
                if y % p == 0:
                    while y % p == 0:
                        y //= p
                    result -= result // p
            if y > 1:
                result -= result // y
            if result in seen:
                valid = False
                break
            seen.add(result)
        
        if not valid:
            # Fallback to a known valid result
            return {'start': 1, 'length': 1}
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
