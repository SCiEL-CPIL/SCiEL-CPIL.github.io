import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute Smallest Prime Factor (SPF) sieve
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
            
    # Localize spf for speed
    spf_local = spf
    
    def get_sigma(n):
        if n == 1:
            return 1
        res = 1
        while n > 1:
            p = spf_local[n]
            exp = 0
            sum_p = 1
            p_pow = 1
            while n % p == 0:
                n //= p
                exp += 1
                p_pow *= p
                sum_p += p_pow
            res *= sum_p
        return res

    best_a = 11
    best_b = 6
    best_err = abs(2*11 - 3*6) / (2*6) # 1/6 ~ 0.166
    
    # 2. Search for k = |2a - 3b|
    max_k_to_check = 100
    
    for k in range(1, max_k_to_check + 1):
        if time.time() > deadline - 5:
            break
            
        b_max = min(N, (2 * N - k) // 3)
        
        if k % 2 == 0:
            if b_max % 2 != 0:
                b_max -= 1
        else:
            if b_max % 2 == 0:
                b_max -= 1
            
        b = b_max
        found = False
        while b >= 1:
            a = (3 * b + k) // 2
            
            if a > N:
                b -= 2
                continue
                
            sa = get_sigma(a)
            sb = get_sigma(b)
            
            if sa == sb:
                err = k / (2 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                found = True
                break
                
            b -= 2
            
            if time.time() > deadline - 2:
                return [int(best_a), int(best_b)]
                
        if found:
            # As reasoned, smaller k with large b is likely optimal.
            # We can break if k is small enough that even the worst case for k+1 is worse.
            # But let's just break on first find for simplicity and speed, 
            # given the monotonicity argument.
            break

    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
