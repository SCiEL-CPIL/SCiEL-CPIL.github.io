import time
import math

def construct():
    deadline = time.time() + 145.0
    
    def sieve_primes(limit):
        """Sieve of Eratosthenes up to limit."""
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return is_prime
    
    def compute_phi_segment(start, end, primes):
        """Compute phi for all numbers in [start, end] using segmented sieve."""
        n = end - start + 1
        phi = list(range(start, end + 1))
        remaining = list(range(start, end + 1))
        
        for p in primes:
            if p > end:
                break
            first = ((start + p - 1) // p) * p
            for j in range(first, end + 1, p):
                idx = j - start
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def find_longest_distinct_run(phi_values, max_len=200):
        """Find longest contiguous subarray with all distinct values."""
        n = len(phi_values)
        best_start = 0
        best_len = 1
        left = 0
        seen = {}
        
        for right in range(n):
            v = phi_values[right]
            if v in seen and seen[v] >= left:
                left = seen[v] + 1
            seen[v] = right
            
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left
            if best_len >= max_len:
                break
        
        return best_start, min(best_len, max_len)
    
    # Generate primes up to sqrt(10^12) = 10^6
    prime_limit = 1000000
    is_prime = sieve_primes(prime_limit)
    primes = [i for i in range(2, prime_limit + 1) if is_prime[i]]
    
    best_overall = (1, 1)  # (length, start)
    
    # Strategy 1: Scan from small numbers with moderate segments
    segment_size = 100000
    search_end = min(10000000, 10**12 - 200)
    
    current_start = 1
    while current_start <= search_end and time.time() < deadline - 5:
        current_end = min(current_start + segment_size - 1, search_end)
        
        phi_vals = compute_phi_segment(current_start, current_end, primes)
        
        rel_start, rel_len = find_longest_distinct_run(phi_vals, 200)
        abs_start = current_start + rel_start
        
        if rel_len > best_overall[0]:
            best_overall = (rel_len, abs_start)
        
        current_start = current_end + 1
    
    # Strategy 2: If time remains, try different ranges
    if time.time() < deadline - 10 and best_overall[0] < 200:
        # Try larger gaps in search
        segment_size2 = 50000
        for base in range(10000000, min(50000000, 10**12 - 200), segment_size2 * 10):
            if time.time() > deadline - 5:
                break
            end2 = min(base + segment_size2 - 1, 10**12 - 200)
            phi_vals2 = compute_phi_segment(base, end2, primes)
            rs, rl = find_longest_distinct_run(phi_vals2, 200)
            if rl > best_overall[0]:
                best_overall = (rl, base + rs)
    
    return {'start': best_overall[1], 'length': best_overall[0]}
