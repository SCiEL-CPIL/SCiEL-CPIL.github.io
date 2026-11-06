import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Sieve of Eratosthenes to find primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_sieve(limit):
        """
        Compute phi(n) for all n in [1, limit] using a linear-like sieve approach.
        Returns a list phi[0..limit] where phi[n] is Euler's totient of n.
        phi[0] = 0.
        """
        phi = list(range(limit + 1))
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi[j] -= phi[j] // i
        return phi

    def find_best_run(phi_vals, offset=0):
        """
        Find the longest run of consecutive distinct phi values.
        phi_vals is a list where phi_vals[i] corresponds to number (offset + i).
        Returns (max_length, best_start_number).
        """
        best_k = 0
        best_start = offset
        seen = set()
        run_start = 0
        run_len = 0

        for i, v in enumerate(phi_vals):
            if v not in seen:
                seen.add(v)
                run_len += 1
            else:
                if run_len > best_k:
                    best_k = run_len
                    best_start = offset + run_start
                # Reset seen, but keep the current value as the start of the new run
                seen = {v}
                run_start = i
                run_len = 1

        if run_len > best_k:
            best_k = run_len
            best_start = offset + run_start

        return best_k, best_start

    def phi_trial(n, primes):
        """Compute phi(n) using trial division by primes."""
        result = n
        y = n
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

    def check_window_large(start, length, primes):
        """
        Check a window of consecutive integers starting at 'start' for distinct phi values.
        Returns the length of the longest distinct prefix.
        """
        seen = set()
        k = 0
        for i in range(length):
            n = start + i
            if n <= 0:
                continue
            v = phi_trial(n, primes)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    def segmented_phi(start, end, primes):
        """
        Compute phi(n) for n in [start, end] using a segmented approach.
        Returns a list of phi values for n = start, start+1, ..., end.
        """
        size = end - start + 1
        if size <= 0:
            return []
        
        # Initialize: phi[n] starts as n
        phi_seg = list(range(start, end + 1))
        
        # For each prime p, find multiples of p in [start, end]
        for p in primes:
            if p * p > end and p > end:
                break
            # Find the first multiple of p >= start
            first_mult = ((start + p - 1) // p) * p
            if first_mult < p:
                first_mult = p  # p itself
            
            # Process all multiples of p in [start, end]
            for j in range(first_mult, end + 1, p):
                idx = j - start
                phi_seg[idx] -= phi_seg[idx] // p
        
        return phi_seg

    # Precompute primes for trial division and segmented sieve
    # We need primes up to sqrt(10^12) = 10^6 for trial division
    primes_for_trial = sieve_primes(10**6)
    
    # For segmented sieve, we need primes up to sqrt(end) for each segment
    # But we can use the same primes list since we only need primes up to sqrt(10^12)
    
    best_k = 1
    best_start = 1

    # Phase 1: Sieve Search from 1 up to a large limit
    # Start with a safe limit and expand
    sieve_limits = [10**6, 5 * 10**6, 2 * 10**7, 5 * 10**7, 10**8]
    
    for limit in sieve_limits:
        if time_remaining() < 10.0:
            break
        
        try:
            phi_vals = compute_phi_sieve(limit)
            # find_best_run expects values for n=offset+i
            # Here offset=1, so phi_vals[1] corresponds to n=1.
            k, s = find_best_run(phi_vals[1:], offset=1)
            if k > best_k:
                best_k = k
                best_start = s
            
        except MemoryError:
            break
        except Exception:
            continue

    # Phase 2: Segmented sieve search in larger ranges
    # Search in ranges like [10^8, 10^9], [10^9, 10^10], etc.
    if time_remaining() > 15.0:
        # Use segmented phi computation
        # Search in chunks to manage memory
        chunk_size = 5 * 10**6
        
        # Define search ranges to explore
        search_ranges = [
            (10**8, 10**8 + chunk_size * 10),
            (10**9, 10**9 + chunk_size * 10),
            (5 * 10**9, 5 * 10**9 + chunk_size * 10),
            (10**10, 10**10 + chunk_size * 10),
            (5 * 10**10, 5 * 10**10 + chunk_size * 10),
            (10**11, 10**11 + chunk_size * 10),
            (5 * 10**11, 5 * 10**11 + chunk_size * 10),
            (10**12 - chunk_size * 10, 10**12),
        ]
        
        for (range_start, range_end) in search_ranges:
            if time_remaining() < 5.0:
                break
            
            # Process in chunks
            pos = range_start
            while pos <= range_end and time_remaining() > 5.0:
                chunk_end = min(pos + chunk_size - 1, range_end)
                try:
                    phi_chunk = segmented_phi(pos, chunk_end, primes_for_trial)
                    k, s = find_best_run(phi_chunk, offset=pos)
                    if k > best_k:
                        best_k = k
                        best_start = s
                except Exception:
                    pass
                pos = chunk_end + 1

    # Phase 3: Check extension of the best found run using trial division
    if time_remaining() > 5.0 and best_k > 1:
        # Try to extend the best run
        # Check a window around the best start
        check_len = best_k + 30
        
        # Try shifting start
        for delta in range(-10, 11):
            if time_remaining() < 1.0:
                break
            s_try = best_start + delta
            if s_try < 1 or s_try + check_len > 10**12:
                continue
            
            k_try = check_window_large(s_try, check_len, primes_for_trial)
            if k_try > best_k:
                best_k = k_try
                best_start = s_try
                check_len = best_k + 30

    # Phase 4: Random Search in Large Range
    if time_remaining() > 5.0:
        random.seed(42)
        
        # Adaptive check length
        check_len = best_k + 20
        samples_done = 0
        max_samples = 1000
        
        # Stratified sampling across different magnitude ranges
        log_ranges = [
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
            (10**9, 10**10),
            (10**10, 10**11),
            (10**11, 10**12),
        ]
        
        while time_remaining() > 2.0 and samples_done < max_samples:
            # Pick a random range
            low_bound, high_bound = random.choice(log_ranges)
            
            if high_bound - low_bound < check_len:
                continue
                
            s = random.randint(low_bound, high_bound - check_len)
            
            k_found = check_window_large(s, check_len, primes_for_trial)
            
            if k_found > best_k:
                best_k = k_found
                best_start = s
                check_len = best_k + 20
            
            samples_done += 1

    # Phase 5: Final local refinement
    if time_remaining() > 1.0 and best_k > 1:
        # Try to extend the best found run further
        # Check both directions
        for delta in range(-20, 21):
            if time_remaining() < 0.5:
                break
            s_try = best_start + delta
            if s_try < 1:
                continue
            if s_try + best_k + 15 > 10**12:
                continue
            
            k_try = check_window_large(s_try, best_k + 15, primes_for_trial)
            if k_try > best_k:
                best_k = k_try
                best_start = s_try

    # Ensure constraints
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    if best_k < 1:
        best_k = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
