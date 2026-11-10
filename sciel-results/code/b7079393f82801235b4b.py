import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    MAX_N = 10000000
    
    # 1. Precompute sigma(n) using NumPy sieve
    sigma_arr = np.zeros(MAX_N + 1, dtype=np.int64)
    
    t0 = time.time()
    for i in range(1, MAX_N + 1):
        sigma_arr[i::i] += i
        # Monitor time during sieve
        if i % 1000000 == 0:
            if time.time() > deadline - 10:
                # If sieve is too slow, we can't finish.
                # But we need sigma.
                # Let's just finish the sieve, it's usually fast.
                pass

    if time.time() > deadline - 5:
        return [11, 6]

    # 2. Search
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5) # ~0.333
    
    # Iterate b from MAX_N down to 1
    # We can convert sigma_arr to a list for faster access?
    # No, numpy array indexing is fast enough for O(1) access in Python loop.
    # Actually, list of ints is faster in CPython for single element access?
    # Let's try converting to list if it's not too memory intensive.
    # 10^7 ints in Python list is ~80MB. Acceptable.
    sigma_list = sigma_arr.tolist()
    
    for b in range(MAX_N, 1, -1):
        if time.time() > deadline - 2:
            break
            
        # Pruning: min possible error for this b is 1/(2b) (for k=1)
        # If 1/(2b) >= best_err, no smaller b can improve
        if 1.0 / (2.0 * b) >= best_err:
            break
            
        # Max k to check such that k/(2b) < best_err
        k_limit = int(2.0 * b * best_err)
        
        # Cap k_limit to prevent excessive work if best_err is still large
        # If k_limit is very large, it means best_err is large.
        # We will find a better best_err quickly if solutions exist.
        # But to be safe, cap at 50000?
        if k_limit > 50000:
            k_limit = 50000
            
        # Check k from 1 to k_limit
        # We check both a = (3b+k)/2 and a = (3b-k)/2
        # (3b+k) and (3b-k) have opposite parity?
        # 3b+k and 3b-k differ by 2k, so same parity.
        # So if one is even, the other is even.
        
        found = False
        for k in range(1, k_limit + 1):
            num = 3 * b + k
            if num % 2 == 0:
                a = num // 2
                if 1 <= a <= MAX_N:
                    if sigma_list[a] == sigma_list[b]:
                        err = k / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                        found = True
                        break # Smallest k is best for this b
                
                # Check the other side
                # a2 = (3b - k) / 2
                # Since num was even, 3b-k is also even.
                a2 = (3 * b - k) // 2
                if 1 <= a2 <= MAX_N and a2 != a:
                    if sigma_list[a2] == sigma_list[b]:
                        err = k / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a2
                            best_b = b
                        found = True
                        break
            else:
                # num is odd, so 3b-k is also odd. No integer a.
                pass
                
        # If we didn't find a match in the first k_limit, we might miss a better one
        # But we capped k_limit. If we capped, we might need to check more?
        # No, if we capped, we just didn't check all.
        # But as best_err improves, k_limit shrinks.
        
    return [int(best_a), int(best_b)]
