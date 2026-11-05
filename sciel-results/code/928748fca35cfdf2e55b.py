"""Program to find long runs of consecutive integers with distinct totients up to 10^12."""
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # 1. Generate primes up to 10^6 (sqrt of 10^12)
    limit = 10**6
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_window(start, length):
        """Compute phi for a window of consecutive integers using segmented sieve approach."""
        # Initialize with the values themselves
        vals = np.arange(start, start + length, dtype=np.int64).astype(np.float64)
        
        # For each prime p, multiply by (1 - 1/p) for numbers divisible by p
        for p in primes:
            if p * p > start + length - 1:
                break
            # Find indices in the window divisible by p
            first = start - (start % p)
            if first < start:
                first += p
            indices = np.arange(first, start + length, p) - start
            if len(indices) > 0:
                vals[indices] *= (1.0 - 1.0/p)
        
        # Handle remaining prime factors
        # For each number, if it has a prime factor > 10^6, we need to handle it
        # But since we're working with numbers up to 10^12, and we've removed all prime factors
        # up to 10^6, any remaining factor > 1 must be prime (since 10^6 * 10^6 = 10^12)
        
        result = np.zeros(length, dtype=np.int64)
        for i in range(length):
            n = start + i
            # Compute phi using the float approximation, then correct
            phi_val = vals[i]
            # The float computation might have rounding errors, so we do exact computation
            # for verification/correction
            temp = n
            phi_exact = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    phi_exact -= phi_exact // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                phi_exact -= phi_exact // temp
            result[i] = phi_exact
        
        return result
    
    def find_longest_run(phi_vals, offset=0):
        """Find the longest run of distinct values in phi_vals."""
        best_len = 1
        best_start = 0
        seen = set()
        current_len = 0
        current_start = 0
        
        for i, v in enumerate(phi_vals):
            if v not in seen:
                if current_len == 0:
                    current_start = i
                current_len += 1
                seen.add(v)
                if current_len > best_len:
                    best_len = current_len
                    best_start = current_start
            else:
                # Collision - restart
                seen = set()
                current_len = 0
                current_start = i + 1
        
        return best_start + offset, best_len
    
    best_start = 1
    best_len = 1
    
    # Strategy: Try several windows at different locations near 10^12
    # Large numbers have more distinct totients due to larger prime factor variations
    
    # First, try a large window near 10^12
    window_size = 100000  # 100k numbers
    
    # Try several starting points
    candidate_starts = []
    
    # Near 10^12
    base = 10**12 - window_size - 1000
    for i in range(20):
        candidate_starts.append(base + i * 10000)
    
    # Some random large starts
    random.seed(42)
    for _ in range(30):
        s = random.randint(10**11, 10**12 - window_size)
        candidate_starts.append(s)
    
    # Also try some medium-sized numbers
    for _ in range(10):
        s = random.randint(10**8, 10**11 - window_size)
        candidate_starts.append(s)
    
    candidate_starts = sorted(set(candidate_starts))
    
    for start in candidate_starts:
        if time.time() > deadline - 5:
            break
        
        if start + window_size - 1 > 10**12:
            continue
        
        # Compute phi for this window
        phi_vals = compute_phi_window(start, window_size)
        
        # Find longest run
        run_start, run_len = find_longest_run(phi_vals, offset=start)
        
        if run_len > best_len:
            best_len = run_len
            best_start = run_start
    
    # If we have time, try to extend the best run found
    if time.time() < deadline - 10 and best_len > 1:
        # Try to extend left and right
        extended_start = best_start
        extended_len = best_len
        
        # Extend left
        try_left = best_start - 1
        while try_left >= 1 and extended_len < 200:
            if time.time() > deadline - 5:
                break
            phi_val = compute_phi_window(try_left, 1)[0]
            # Check if this value collides with existing run
            existing_vals = set(compute_phi_window(best_start, extended_len))
            if phi_val not in existing_vals:
                extended_start = try_left
                extended_len += 1
                try_left -= 1
            else:
                break
        
        # Extend right
        try_right = best_start + extended_len
        while try_right <= 10**12 and extended_len < 200:
            if time.time() > deadline - 5:
                break
            phi_val = compute_phi_window(try_right, 1)[0]
            existing_vals = set(compute_phi_window(extended_start, extended_len))
            if phi_val not in existing_vals:
                extended_len += 1
                try_right += 1
            else:
                break
        
        if extended_len > best_len:
            best_len = extended_len
            best_start = extended_start
    
    # Cap at 200
    if best_len > 200:
        best_len = 200
    
    return {'start': best_start, 'length': best_len}
