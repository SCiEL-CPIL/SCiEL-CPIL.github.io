import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 10000000

    # Compute sigma(n) for all n up to N using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)

    for i in range(1, N + 1):
        if time.time() > deadline - 15.0:
            # If sieve is too slow, use fallback
            return [11, 6]
        sigma[i::i] += i

    if time.time() > deadline - 10.0:
        return [11, 6]

    # Search for best pair
    # Error = |k| / (2b) where k = 2a - 3b
    # We want to minimize this.
    # Iterate k from small |k| upwards.
    # For each k, find largest b such that a = (3b+k)//2 is valid and sigma(a)==sigma(b).
    
    best_err_num = None
    best_err_den = None
    best_a = None
    best_b = None
    
    # We'll check k values in order of |k|: 1, -1, 2, -2, 3, -3, ...
    # For each k, the error for a valid b is |k|/(2b)
    # We can stop when |k|/(2*1) >= best_err, i.e., |k| >= 2 * best_err_num * N / best_err_den
    
    max_k_abs = 200000  # Upper bound for |k| to check
    
    for k_abs in range(1, max_k_abs + 1):
        if time.time() > deadline - 5.0:
            break
            
        for k in [k_abs, -k_abs]:
            if time.time() > deadline - 5.0:
                break
                
            # Check if we can stop early
            if best_err_num is not None:
                # Check if |k| / (2*1) >= best_err_num / best_err_den
                # |k| * best_err_den >= best_err_num * 2
                # But we want the minimum possible error for this k, which is |k|/(2*N)
                # So if |k|/(2*N) >= best_err, we can skip
                # |k| * best_err_den >= best_err_num * 2 * N
                if abs(k) * best_err_den >= best_err_num * 2 * N:
                    break  # Both k and -k have same |k|, so break outer too
            
            # Determine valid range for b
            # a = (3b + k) / 2 must be integer and in [1, N]
            # 3b + k must be even: b % 2 == k % 2 (since 3b ≡ b mod 2)
            # 1 <= (3b + k)/2 <= N  =>  2 <= 3b + k <= 2N
            # 2 - k <= 3b <= 2N - k
            
            b_min_val = 2 - k
            b_max_val = 2 * N - k
            
            # b >= ceil((2-k)/3), b <= floor((2N-k)/3)
            # Ceiling division for possibly negative numerator:
            # ceil(x/3) = -((-x)//3)
            b_min = -((-b_min_val) // 3)
            b_max = b_max_val // 3
            
            # Clamp to [1, N]
            if b_min < 1:
                b_min = 1
            if b_max > N:
                b_max = N
            
            # Parity check: b % 2 == k % 2
            parity = k & 1  # k mod 2
            if b_min & 1 != parity:
                b_min += 1
            if b_max & 1 != parity:
                b_max -= 1
                
            if b_min > b_max:
                continue
            
            # Iterate b from b_max down to b_min (step 2 to maintain parity)
            # We want the largest b to minimize error for this k
            for b in range(b_max, b_min - 1, -2):
                a = (3 * b + k) // 2
                # a should be in [1, N] by construction, but double-check
                if 1 <= a <= N:
                    if sigma[a] == sigma[b]:
                        # Found a match
                        err_num = abs(k)
                        err_den = 2 * b
                        
                        # Compare err_num/err_den with best_err_num/best_err_den
                        if best_err_num is None or err_num * best_err_den < best_err_num * err_den:
                            best_err_num = err_num
                            best_err_den = err_den
                            best_a = a
                            best_b = b
                        break  # Found best b for this k (largest b)
                    
    if best_a is not None:
        return [int(best_a), int(best_b)]
        
    # Fallback
    return [11, 6]
