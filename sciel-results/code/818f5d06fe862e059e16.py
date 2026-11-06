import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s margin
    
    # 1. Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    max_n = 10**12
    segment_size = 10**6
    
    best_k = 1
    best_s = 1
    
    # Helper to find longest distinct run in an array
    def find_longest_distinct(arr):
        if len(arr) == 0:
            return 0, 0
        last_seen = {}
        start = 0
        max_len = 0
        max_start = 0
        for i in range(len(arr)):
            val = arr[i]
            if val in last_seen and last_seen[val] >= start:
                start = last_seen[val] + 1
            last_seen[val] = i
            if i - start + 1 > max_len:
                max_len = i - start + 1
                max_start = start
        return max_len, max_start

    # Process segments
    current_start = 1
    while current_start + segment_size <= max_n:
        if time.time() > deadline:
            break
            
        current_end = current_start + segment_size - 1
        if current_end > max_n:
            current_end = max_n
            current_start = current_end - segment_size + 1
            if current_start < 1:
                current_start = 1
        
        n_seg = current_end - current_start + 1
        
        # Initialize arrays
        phi_arr = np.arange(current_start, current_end + 1, dtype=np.int64)
        temp_arr = np.arange(current_start, current_end + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > current_end:
                break
                
            first = ((current_start + p - 1) // p) * p
            if first < current_start:
                first = current_start
                
            idxs = np.arange(first, current_end + 1, p) - current_start
            
            while True:
                mask = temp_arr[idxs] % p == 0
                if not np.any(mask):
                    break
                valid_idxs = idxs[mask]
                phi_arr[valid_idxs] -= phi_arr[valid_idxs] // p
                temp_arr[valid_idxs] //= p
                idxs = valid_idxs
                
        mask = temp_arr > 1
        if np.any(mask):
            phi_arr[mask] -= phi_arr[mask] // temp_arr[mask]
            
        k_seg, s_seg = find_longest_distinct(phi_arr)
        
        if k_seg > best_k:
            best_k = k_seg
            best_s = current_start + s_seg
            
        current_start += segment_size
        
    return {'start': best_s, 'length': best_k}
