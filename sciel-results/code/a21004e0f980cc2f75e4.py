import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def compute_phi_block(start, length):
        """
        Computes phi(n) for n in [start, start+length-1] using segmented sieve.
        Returns a list of length 'length' containing phi values.
        """
        phi_vals = [0] * length
        rem_vals = [0] * length
        for i in range(length):
            n = start + i
            phi_vals[i] = n
            rem_vals[i] = n
            
        # Sieve with primes
        for p in primes:
            if p * p > start + length - 1:
                break
            # Find the first multiple of p in the range
            first_multiple = ((start + p - 1) // p) * p
            for j in range(first_multiple - start, length, p):
                if phi_vals[j] % p == 0:
                    phi_vals[j] -= phi_vals[j] // p
                while rem_vals[j] % p == 0:
                    rem_vals[j] //= p
        
        # Handle remaining prime factors
        for i in range(length):
            if rem_vals[i] > 1:
                phi_vals[i] -= phi_vals[i] // rem_vals[i]
                
        return phi_vals

    def find_best_run_in_block(block_start, block_len):
        """
        Finds the longest run of distinct phi values in the block.
        Returns (best_k, best_start_offset).
        """
        if time.time() > deadline:
            return 0, 0
            
        phis = compute_phi_block(block_start, block_len)
        
        max_k = 0
        max_start = 0
        seen = set()
        cur_k = 0
        cur_start = 0
        
        for i in range(block_len):
            if phis[i] not in seen:
                if cur_k == 0:
                    cur_start = i
                cur_k += 1
                seen.add(phis[i])
                if cur_k > max_k:
                    max_k = cur_k
                    max_start = cur_start
            else:
                # Collision, reset
                seen = set()
                cur_k = 0
                
        return max_k, max_start

    best_k = 1
    best_s = 1
    
    block_size = 10**5
    
    # Search in high ranges
    # We will scan contiguous regions with overlapping blocks to catch boundary crossings
    # Step size = block_size - 200
    
    search_regions = [
        (10**12 - 10**7, 10**7),
        (10**12 - 2*10**7, 10**7),
        (10**11, 10**7),
        (5*10**11, 10**7)
    ]
    
    for region_start, region_len in search_regions:
        if time.time() > deadline:
            break
        # Step through the region with overlap
        step = block_size - 200
        for offset in range(0, region_len - 200, step):
            if time.time() > deadline:
                break
            block_start = region_start + offset
            # Ensure block doesn't exceed 10^12
            max_block_len = 10**12 - block_start + 1
            if max_block_len < 200:
                break
            block_len = min(block_size, max_block_len)
            
            k, offset_in_block = find_best_run_in_block(block_start, block_len)
            if k > best_k:
                best_k = k
                best_s = block_start + offset_in_block
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}

    # Random search if needed
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 100:
        attempts += 1
        s = random.randint(10**11, 10**12 - 10**5)
        block_len = min(block_size, 10**12 - s + 1)
        if block_len < 200: continue
        k, offset_in_block = find_best_run_in_block(s, block_len)
        if k > best_k:
            best_k = k
            best_s = s + offset_in_block
            if best_k >= 200:
                break

    # Final validation
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
    if best_k < 1:
        best_k = 1
        best_s = 1

    return {'start': best_s, 'length': best_k}
