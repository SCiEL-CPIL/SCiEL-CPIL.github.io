import time
import math
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Precompute small primes up to 1e6 for fast factorization
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(1000000)
    
    def phi_fast(x):
        """Compute Euler's totient using precomputed primes."""
        if x <= 0:
            return 0
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

    def segmented_phi(L, R):
        """
        Compute phi for all n in [L, R] using segmented sieve approach.
        Returns list of phi values for n = L, L+1, ..., R.
        """
        size = R - L + 1
        # Initialize phi[n] = n+1 for n in range
        phi_arr = list(range(L, R + 1))
        rem_arr = list(range(L, R + 1))
        
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start < p * p:
                start = p * p
            if start > R:
                continue
            for m in range(start, R + 1, p):
                idx = m - L
                if rem_arr[idx] % p == 0:
                    phi_arr[idx] -= phi_arr[idx] // p
                    while rem_arr[idx] % p == 0:
                        rem_arr[idx] //= p
        
        # Handle remaining prime factors
        for i in range(size):
            if rem_arr[i] > 1:
                phi_arr[i] -= phi_arr[i] // rem_arr[i]
        
        return phi_arr

    def find_best_run_in_vals(vals, offset):
        """
        Find the longest run of consecutive distinct values in vals.
        Returns (length, start_index_in_vals).
        """
        n = len(vals)
        if n == 0:
            return 0, 0
        best_k = 1
        best_s = 0
        seen = set()
        current_k = 1
        seen.add(vals[0])
        
        for i in range(1, n):
            if vals[i] in seen:
                # Reset run
                if current_k > best_k:
                    best_k = current_k
                    best_s = i - current_k
                # Start new run
                seen = {vals[i]}
                current_k = 1
            else:
                seen.add(vals[i])
                current_k += 1
                # Update best if current run is longer and we have enough room
                # Note: We want the run to be within the array.
                if current_k > best_k:
                    best_k = current_k
                    best_s = i - current_k + 1
        
        # Check the last run
        if current_k > best_k:
            best_k = current_k
            best_s = n - current_k
            
        return best_k, best_s

    best_k = 1
    best_s = 1
    
    def update_best(start, k):
        nonlocal best_k, best_s
        if k > best_k:
            best_k = k
            best_s = start

    # Strategy 1: Dense scan of small ranges using segmented phi
    # Scan [1, 100000] first - this is very fast with segmented approach
    if time.time() < deadline:
        L, R = 1, 100000
        vals = segmented_phi(L, R)
        k, idx = find_best_run_in_vals(vals, L)
        update_best(L + idx, k)

    # Strategy 2: Scan medium ranges in chunks
    # Each chunk of 100000 is fast; scan up to ~10 million total
    chunk_size = 100000
    max_scan = 10_000_000
    if time.time() < deadline - 20:
        for L in range(100001, max_scan + 1, chunk_size):
            if time.time() > deadline - 20:
                break
            R = min(L + chunk_size - 1, max_scan)
            vals = segmented_phi(L, R)
            k, idx = find_best_run_in_vals(vals, L)
            update_best(L + idx, k)

    # Strategy 3: Random sampling with local dense scan using segmented phi
    attempts = 0
    while time.time() < deadline - 5 and attempts < 5000:
        attempts += 1
        # Sample from various ranges
        r = random.random()
        if r < 0.15:
            s = random.randint(10**6, 10**8)
        elif r < 0.30:
            s = random.randint(10**8, 10**10)
        elif r < 0.45:
            s = random.randint(10**10, 10**11)
        elif r < 0.60:
            s = random.randint(10**11, 10**12 - 200)
        else:
            s = random.randint(10**5, 10**12 - 200)
            
        if s < 1 or s + 200 > 10**12:
            continue
        
        # Quick check: compute phi for 200 values starting at s
        # Use segmented phi for a small window
        win_L = max(1, s - 50)
        win_R = min(10**12, s + 250)
        vals = segmented_phi(win_L, win_R)
        
        # Find best run in this window
        k_local, idx_local = find_best_run_in_vals(vals, win_L)
        start_local = win_L + idx_local
        
        # Ensure the run is fully within bounds
        if start_local + k_local - 1 <= 10**12:
            update_best(start_local, k_local)
        else:
            # Truncate if necessary
            valid_k = 10**12 - start_local + 1
            if valid_k > 0 and valid_k < k_local:
                update_best(start_local, valid_k)

    # Ensure validity
    s = best_s
    k = best_k
    if s < 1:
        s = 1
    if s + k - 1 > 10**12:
        k = 10**12 - s + 1
        if k < 1:
            s = 1
            k = 1
    if k > 200:
        k = 200
    if k < 1:
        k = 1
    
    return {'start': s, 'length': k}
