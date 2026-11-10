import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Sieve primes up to sqrt(10^12) = 10^6
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = 0
        if limit >= 1:
            is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(10**6)
    primes_arr = np.array(primes, dtype=np.int64)
    
    def compute_phi_segmented(start, length):
        """Compute phi for numbers from start to start+length-1 using segmented approach."""
        if time_left() < 2:
            return None
        
        # For each number, we'll factor it using primes up to sqrt(max_value)
        # max value is start + length - 1
        max_val = start + length - 1
        sqrt_max = int(math.isqrt(max_val)) + 1
        
        # Create arrays for tracking
        # For each number in range, store remaining value and phi multiplier
        remaining = np.arange(start, start + length, dtype=np.int64)
        phi_vals = remaining.copy()
        
        # Sieve with each prime
        for p in primes:
            if p > sqrt_max:
                break
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            
            for n in range(first, start + length, p):
                idx = n - start
                if remaining[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # For any remaining value > 1, it's a prime factor
        for i in range(length):
            if remaining[i] > 1:
                phi_vals[i] = phi_vals[i] // remaining[i] * (remaining[i] - 1)
        
        return phi_vals
    
    def find_longest_run(phi_vals, start):
        """Find the longest run of consecutive distinct phi values."""
        if len(phi_vals) < 1:
            return 1, start
        
        best_len = 1
        best_start = start
        seen = set()
        window_start = 0
        
        for i in range(len(phi_vals)):
            val = phi_vals[i]
            if val in seen:
                # Shrink window from left until val is removed
                while window_start < i and phi_vals[window_start] != val:
                    seen.discard(phi_vals[window_start])
                    window_start += 1
                if window_start == i:
                    # This means we need to skip this element
                    window_start = i + 1
                    seen.clear()
                else:
                    seen.discard(val)
            else:
                seen.add(val)
            
            current_len = i - window_start + 1
            if current_len > best_len:
                best_len = current_len
                best_start = start + window_start
        
        return best_len, best_start
    
    best_k = 1
    best_s = 1
    
    # Strategy: search in chunks, starting from smaller ranges and going up
    # For each range, compute phi and find longest run
    
    # First, try a medium range to get a quick result
    test_ranges = [
        (1, 50000),
        (100000, 50000),
        (1000000, 100000),
        (10000000, 100000),
        (100000000, 200000),
        (1000000000, 200000),
        (10000000000, 200000),
        (100000000000, 200000),
        (500000000000, 200000),
        (900000000000, 200000),
    ]
    
    for range_start, range_len in test_ranges:
        if time_left() < 5:
            break
        
        if range_start + range_len - 1 > 10**12:
            continue
        
        phi_vals = compute_phi_segmented(range_start, range_len)
        if phi_vals is None:
            continue
        
        k, s = find_longest_run(phi_vals, range_start)
        
        if k > best_k:
            best_k = k
            best_s = s
        
        # If we found a good result, we can try to extend
        if best_k >= 50:
            # Try searching around the best found location
            if time_left() < 10:
                break
    
    # If we haven't found great results, try a more targeted search
    if best_k < 20 and time_left() > 20:
        # Try searching in smaller chunks at various locations
        search_locations = [
            10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 5*10**11, 9*10**11
        ]
        chunk_size = 50000
        
        for loc in search_locations:
            if time_left() < 5:
                break
            if loc + chunk_size > 10**12:
                continue
            
            phi_vals = compute_phi_segmented(loc, chunk_size)
            if phi_vals is None:
                continue
            
            k, s = find_longest_run(phi_vals, loc)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Final verification: check that our answer is valid
    # Recompute phi for the claimed range to verify
    if best_k > 1 and time_left() > 5:
        verify_start = best_s
        verify_len = best_k
        phi_check = compute_phi_segmented(verify_start, verify_len)
        if phi_check is not None:
            # Check pairwise distinctness
            seen = set()
            valid = True
            for v in phi_check:
                if v in seen:
                    valid = False
                    break
                seen.add(v)
            if not valid:
                # Fall back to a safe answer
                best_k = 1
                best_s = 1
    
    return {'start': best_s, 'length': best_k}
