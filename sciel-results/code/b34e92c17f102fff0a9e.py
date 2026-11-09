import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Step 1: Sieve primes up to some limit for segmented phi computation
    # We need primes up to sqrt(max_n) for factoring, but for segmented phi,
    # we need primes up to the segment limit.
    
    # Actually, for segmented phi sieve:
    # - We have a segment [lo, hi].
    # - For each prime p <= hi, we mark multiples of p in [lo, hi] and update phi.
    # - We need primes up to hi, which is too large if hi is 10^9.
    
    # Better approach:
    # - Sieve primes up to sqrt(hi) for factoring individual numbers.
    # - But that's slow for large ranges.
    
    # Alternative: Use the standard segmented phi sieve:
    # - Sieve primes up to hi (impossible for hi = 10^9).
    # - Instead, sieve primes up to some limit L, and for numbers with factors > L,
    #   handle them separately.
    
    # Actually, the standard approach for computing phi(n) for n in [lo, hi]:
    # 1. Create an array phi[0..(hi-lo)] initialized to lo, lo+1, ..., hi.
    # 2. Create an array is_prime_flag or use primes up to sqrt(hi).
    # 3. For each prime p <= sqrt(hi):
    #    - For each multiple m of p in [lo, hi]:
    #      - phi[m - lo] -= phi[m - lo] // p
    # 4. For each number n in [lo, hi], if the remaining factor (after dividing by all primes <= sqrt(hi)) is > 1, it's a prime factor, so phi[n-lo] -= phi[n-lo] // remaining_factor.
    
    # This works but step 3 requires primes up to sqrt(hi). For hi = 10^9, sqrt(hi) = 31623, which is manageable.
    
    # Let me implement this.
    
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(limit + 1) if is_prime[i]]
    
    # Determine search range
    # We'll search in segments. Let's try to search up to 10^8 or 10^9.
    
    # First, let's do a quick search in [1, 100000] to establish a baseline.
    # Then extend to larger ranges.
    
    # Sieve primes up to sqrt(10^9) = 31623
    max_search = 10**9
    prime_limit = int(math.isqrt(max_search)) + 1
    primes = sieve_primes(prime_limit)
    
    best_k = 0
    best_s = 1
    
    def compute_phi_segment(lo, hi):
        """Compute phi(n) for n in [lo, hi] using segmented sieve."""
        n = hi - lo + 1
        phi = list(range(lo, hi + 1))
        
        for p in primes:
            if p * p > hi:
                break
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            for m in range(start, hi + 1, p):
                phi[m - lo] -= phi[m - lo] // p
        
        # Handle remaining prime factors
        for i in range(n):
            val = phi[i]
            # The remaining factor after dividing by all primes <= sqrt(hi)
            # Actually, we need to track the remaining part.
            # Hmm, this approach is flawed. Let me use a different method.
            pass
        
        return phi
    
    # The above is flawed because we don't track the remaining factor.
    # Let me use a different approach: for each number, factor it using primes up to sqrt(n).
    
    # Actually, for a segmented approach, the standard method is:
    # 1. phi[0..n-1] = [lo, lo+1, ..., hi]
    # 2. For each prime p <= hi:
    #    - For each multiple m of p in [lo, hi]:
    #      - phi[m-lo] = phi[m-lo] // p * (p-1)
    # 3. This requires primes up to hi, which is too large.
    
    # Alternative: Use the fact that for n in [lo, hi], we can factor n using primes up to sqrt(hi).
    # After dividing out all prime factors <= sqrt(hi), if the remaining is > 1, it's a prime factor.
    
    # Let me implement a direct phi computation using precomputed primes.
    
    def phi_direct(n, primes):
        """Compute phi(n) using precomputed primes up to sqrt(n)."""
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result
    
    # This is O(sqrt(n)/log(sqrt(n))) per number, which is slow for large ranges.
    
    # Better: Use the segmented sieve properly.
    # The key insight: for the segmented phi sieve, we only need to process primes p where p <= hi.
    # But we can limit to p <= sqrt(hi) for the main loop, and handle the rest.
    
    # Actually, let me use a different strategy:
    # - Search in the first, say, 5 * 10^7 numbers.
    # - Use a direct phi computation with precomputed primes.
    # - This should be fast enough.
    
    # Let me benchmark: phi_direct for n up to 10^7, with primes up to 31623.
    # Each phi computation is O(number of prime factors), which is small on average.
    
    # Let me try searching up to 10^7 first.
    
    def find_longest_run(lo, hi, primes):
        """Find the longest run of consecutive integers in [lo, hi] with distinct phi values."""
        n = hi - lo + 1
        phi_vals = [0] * n
        for i in range(n):
            phi_vals[i] = phi_direct(lo + i, primes)
        
        best_k = 0
        best_s = lo
        seen = set()
        current_k = 0
        current_s = lo
        
        for i in range(n):
            if phi_vals[i] not in seen:
                if current_k == 0:
                    current_s = lo + i
                seen.add(phi_vals[i])
                current_k += 1
            else:
                if current_k > best_k:
                    best_k = current_k
                    best_s = current_s
                # Reset: we need to find the new start
                # The new run starts after the last occurrence of phi_vals[i]
                # But we don't track that. Let's use a different approach.
                # Actually, we can just reset and start fresh from the next position.
                # But this might miss runs that start before i.
                # Better approach: use a sliding window.
                seen.clear()
                current_k = 0
                # We'll restart the window from i+1, but this is O(n^2) worst case.
                # For n = 10^7, this is too slow.
        
        if current_k > best_k:
            best_k = current_k
            best_s = current_s
        
        return best_k, best_s
    
    # The sliding window approach needs to be more careful.
    # Let me use a proper sliding window:
    
    def find_longest_run_v2(lo, hi, primes):
        """Find the longest run of consecutive integers in [lo, hi] with distinct phi values."""
        n = hi - lo + 1
        phi_vals = [0] * n
        for i in range(n):
            phi_vals[i] = phi_direct(lo + i, primes)
        
        best_k = 0
        best_s = lo
        seen = {}
        left = 0
        
        for right in range(n):
            val = phi_vals[right]
            if val in seen:
                # Move left past the previous occurrence
                left = max(left, seen[val] + 1)
            seen[val] = right
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = lo + left
        
        return best_k, best_s
    
    # Now let's search in segments.
    
    # First, search [1, 100000] to establish a baseline.
    seg_size = 100000
    max_n = 10**7  # Search up to 10^7
    
    for seg_start in range(1, max_n + 1, seg_size):
        if time.time() > deadline - 30:
            break
        seg_end = min(seg_start + seg_size - 1, max_n)
        k, s = find_longest_run_v2(seg_start, seg_end, primes)
        if k > best_k:
            best_k = k
            best_s = s
    
    # If we have time, search in larger segments or different ranges.
    if time.time() < deadline - 30 and best_k < 20:
        # Try searching in [10^7, 2*10^7]
        for seg_start in range(10**7, 2*10**7, seg_size):
            if time.time() > deadline - 15:
                break
            seg_end = min(seg_start + seg_size - 1, 2*10**7)
            k, s = find_longest_run_v2(seg_start, seg_end, primes)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Ensure constraints
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    
    return {'start': best_s, 'length': best_k}
