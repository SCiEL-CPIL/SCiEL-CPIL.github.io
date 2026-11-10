"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

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
                step = i
                start = i * i
                is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    # Precompute primes up to 10^6 (sufficient for trial division of numbers up to ~10^12)
    PRIME_LIMIT = 1000000
    primes = sieve_primes(PRIME_LIMIT)
    
    def phi_fast(x):
        """Compute Euler's totient using precomputed primes for trial division."""
        if x == 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def find_longest_run_in_array(phi_vals, max_k=200):
        """Find the longest contiguous subsequence with all distinct values in phi_vals."""
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        
        best_k = 1
        best_idx = 0
        
        # Sliding window approach with a set
        seen = set()
        left = 0
        for right in range(n):
            val = phi_vals[right]
            while val in seen and left <= right:
                seen.remove(phi_vals[left])
                left += 1
            seen.add(val)
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_idx = left
            if best_k >= max_k:
                return max_k, best_idx
        
        return best_k, best_idx
    
    best_k = 1
    best_s = 1
    deadline = _START_TIME + _TIME_LIMIT - 5.0  # 5s safety margin
    
    MAX_K = 200
    
    def search_in_range(start, end, max_k):
        """Search for the longest run of distinct phi values in [start, end]."""
        n = end - start + 1
        if n <= 0:
            return 1, start
        
        # Compute phi values for the entire range
        phi_vals = [0] * n
        for i in range(n):
            if time.time() > deadline:
                break
            phi_vals[i] = phi_fast(start + i)
        
        # Find longest run
        k_found, idx = find_longest_run_in_array(phi_vals, max_k)
        return k_found, start + idx
    
    # Search multiple ranges to cover different parts of the number line
    # Focus on ranges where longer runs are more likely
    ranges_to_try = [
        (1, 50000),
        (50001, 100000),
        (1000000, 1100000),
        (10000000, 10100000),
        (100000000, 100100000),
        (10**9, 10**9 + 100000),
        (10**10, 10**10 + 100000),
        (10**11, 10**11 + 100000),
        (10**12 - 100000, 10**12),
    ]
    
    for start, end in ranges_to_try:
        if time.time() > deadline:
            break
        if end > 10**12:
            end = 10**12
        if start < 1:
            start = 1
        if end < start:
            continue
        k_found, s_found = search_in_range(start, end, MAX_K)
        if k_found > best_k:
            best_k = k_found
            best_s = s_found
        # If we found a long run, try to extend nearby
        if k_found >= 10:
            left_start = max(1, s_found - k_found * 2)
            right_end = min(10**12, s_found + k_found + 500)
            if left_start < s_found or right_end > s_found + k_found - 1:
                extended_k, extended_s = search_in_range(left_start, right_end, MAX_K)
                if extended_k > best_k:
                    best_k = extended_k
                    best_s = extended_s
    
    # If time remains, do a more thorough search in additional ranges
    if time.time() < deadline - 15:
        # Try more random-ish ranges
        additional_ranges = [
            (5 * 10**8, 5 * 10**8 + 100000),
            (2 * 10**10, 2 * 10**10 + 100000),
            (7 * 10**11, 7 * 10**11 + 100000),
            (3 * 10**11, 3 * 10**11 + 100000),
        ]
        for start, end in additional_ranges:
            if time.time() > deadline:
                break
            if end > 10**12:
                end = 10**12
            if start < 1:
                start = 1
            if end < start:
                continue
            k_found, s_found = search_in_range(start, end, MAX_K)
            if k_found > best_k:
                best_k = k_found
                best_s = s_found
            if k_found >= 10:
                left_start = max(1, s_found - k_found * 2)
                right_end = min(10**12, s_found + k_found + 500)
                if left_start < s_found or right_end > s_found + k_found - 1:
                    extended_k, extended_s = search_in_range(left_start, right_end, MAX_K)
                    if extended_k > best_k:
                        best_k = extended_k
                        best_s = extended_s
    
    # Final verification and constraint enforcement
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
        best_s = max(1, best_s + (200 - best_k))
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
