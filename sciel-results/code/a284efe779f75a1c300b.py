import time
import math
import random

def construct():
    # Deadline guard: 120s limit, use 110s margin
    deadline = time.time() + 110
    
    # 1. Sieve primes up to 10^6
    # sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = 0
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Fast phi computation using precomputed primes
    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    # 3. Search for a window of length 200 with distinct phi values
    max_k = 200
    # Search range: [10^11, 10^12 - 200]
    low_start = 10**11
    high_start = 10**12 - max_k
    
    best_s = 1
    best_k = 1
    
    # Strategy: 
    # 1. First, try to find a run of length 200 using random sampling in high ranges.
    # 2. If found, return immediately.
    # 3. If not found within a certain number of attempts or time, fall back to segmented sieve 
    #    to scan larger contiguous blocks to find the best run.
    
    attempts = 0
    found_200 = False
    
    # Phase 1: Random sampling for k=200
    # We expect high probability of success in high ranges.
    while time.time() < deadline - 20 and attempts < 200:
        attempts += 1
        s = random.randint(low_start, high_start)
        
        # Check if phi(s) ... phi(s + max_k - 1) are distinct
        seen = set()
        valid = True
        current_k = 0
        for i in range(max_k):
            n = s + i
            p_val = phi(n)
            if p_val in seen:
                valid = False
                break
            seen.add(p_val)
            current_k = i + 1
        
        if valid:
            return {'start': s, 'length': max_k}
        
        if current_k > best_k:
            best_k = current_k
            best_s = s

    # Phase 2: If we didn't find 200, use segmented sieve to scan larger blocks.
    # This is slower per element but allows checking many elements at once.
    # We'll scan a few large blocks in high ranges.
    
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented sieve."""
        n = hi - lo + 1
        if n <= 0:
            return []
        # We already have primes up to 10^6. 
        # For numbers up to 10^12, sqrt is 10^6, so our primes are sufficient for factorization.
        # However, segmented sieve usually computes phi by iterating primes.
        
        # Initialize phi and remaining
        phi_arr = [lo + i for i in range(n)]
        remaining = [lo + i for i in range(n)]
        
        # Iterate through primes
        for p in primes:
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi_arr[idx] -= phi_arr[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > 10^6
        for i in range(n):
            if remaining[i] > 1:
                phi_arr[i] -= phi_arr[i] // remaining[i]
        
        return phi_arr

    def find_best_run_in_segment(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values in phi_arr."""
        n = len(phi_arr)
        best_k = 0
        best_s = offset
        i = 0
        while i < n:
            seen = set()
            k = 0
            while i + k < n and k < max_len and phi_arr[i + k] not in seen:
                seen.add(phi_arr[i + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = offset + i
            i += 1
        return best_s, best_k

    # Try scanning a few large blocks
    # Block size 10,000 to 100,000
    block_size = 50000
    num_blocks = 5
    
    for b in range(num_blocks):
        if time.time() > deadline - 5:
            break
        
        # Pick a random start for this block in the high range
        # Ensure block fits in [low_start, high_start]
        block_start = random.randint(low_start, high_start - block_size)
        block_end = block_start + block_size - 1
        
        phi_arr = compute_phi_segmented(block_start, block_end)
        
        s, k = find_best_run_in_segment(phi_arr, block_start, max_len=max_k)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k == max_k:
            return {'start': best_s, 'length': best_k}

    # Final check
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1

    return {'start': best_s, 'length': best_k}
