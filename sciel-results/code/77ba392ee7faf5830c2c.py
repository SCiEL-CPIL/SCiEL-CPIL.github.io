import math
import time

def construct():
    start_time = time.time()
    LIMIT = 120.0
    MAX_S = 10**12
    TIME_MARGIN = 5.0

    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Phi calculation for a block
    def compute_phis(start, count):
        phis = []
        for i in range(count):
            n = start + i
            temp = n
            phi_val = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    phi_val -= phi_val // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                phi_val -= phi_val // temp
            phis.append(phi_val)
        return phis

    def find_longest_distinct(phis):
        seen = {}
        l = 0
        max_len = 0
        max_start = 0
        for r in range(len(phis)):
            v = phis[r]
            if v in seen:
                prev = seen[v]
                if prev >= l:
                    l = prev + 1
            seen[v] = r
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start = l
        return max_len, max_start

    # 3. Search
    best_k = 1
    best_s = 1
    
    # We search in blocks of 1000
    block_size = 1000
    current_end = MAX_S
    
    while current_end >= 1:
        if time.time() - start_time > LIMIT - TIME_MARGIN:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        # Compute phis
        phis = compute_phis(block_start, block_size)
        
        # Find longest run
        max_len, max_start_idx = find_longest_distinct(phis)
        
        if max_len > best_k:
            best_k = max_len
            best_s = block_start + max_start_idx
            
        if best_k >= 200:
            return {'start': best_s, 'length': best_k}
            
        # Move to previous block
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
