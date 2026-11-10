import numpy as np
import time

def construct():
    start_time = time.time()
    N = 10000000
    
    # Compute sigma(n) for n = 1..N using additive sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        if d % 100000 == 0:
            if time.time() - start_time > 55:
                break
        sigma[d::d] += d
    
    elapsed = time.time() - start_time
    if elapsed > 60:
        return [11, 6]
    
    # Search by k = |2a - 3b|, prioritizing small k
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    max_k = 1000
    for k in range(1, max_k + 1):
        if time.time() - start_time > 82:
            break
        
        # 2a - 3b = k => a = (3b + k) / 2
        # Need 3b + k even => b ≡ k (mod 2)
        # a <= N => b <= (2N - k) / 3
        # a >= 1 => b >= 1
        
        if k % 2 == 0:
            b_max = min(N, (2 * N - k) // 3)
            if b_max % 2 != 0:
                b_max -= 1
            b_min = 2
        else:
            b_max = min(N, (2 * N - k) // 3)
            if b_max % 2 != 1:
                b_max -= 1
            b_min = 1
        
        if b_max < b_min:
            continue
        
        # Search from largest b down
        for b in range(b_max, b_min - 1, -2):
            if time.time() - start_time > 84:
                break
            a = (3 * b + k) // 2
            if a < 1 or a > N:
                continue
            if sigma[a] == sigma[b]:
                err = k / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = int(a)
                    best_b = int(b)
                break  # Largest b for this k
    
    return [int(best_a), int(best_b)]
