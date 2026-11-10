import numpy as np
import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s limit

    N = 10_000_000

    # Phase 1: Compute sigma using numpy divisor-sum sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 1000000 == 0:
            if time.time() > deadline - 45:
                break

    if time.time() > deadline - 40:
        return [11, 6]

    # Phase 2: Search over small k values in order of |k|
    best_a, best_b = 11, 6
    best_num = 4   # |2*11 - 3*6| = 4
    best_den = 12  # 2*6

    max_k = 5000
    for abs_k in range(1, max_k + 1):
        if time.time() > deadline - 5:
            break
        for k in ([abs_k, -abs_k] if abs_k > 0 else [0]):
            if time.time() > deadline - 5:
                break
            
            # a = (3b + k) / 2, need 3b + k even => b + k even
            min_b = max(1, math.ceil((2 - k) / 3))
            max_b = min(N, math.floor((2 * N - k) / 3))
            
            if min_b > max_b:
                continue
            
            if k % 2 == 0:
                if min_b % 2 != 0:
                    min_b += 1
                step = 2
            else:
                if min_b % 2 == 0:
                    min_b += 1
                step = 2
            
            # Search from largest b downward (minimizes error |k|/(2b))
            checks = 0
            max_checks = 300000
            
            b = max_b
            if b % 2 != (k % 2):
                b -= 1
            
            while b >= min_b and checks < max_checks:
                if time.time() > deadline - 5:
                    break
                checks += 1
                a = (3 * b + k) // 2
                if a < 1 or a > N:
                    b -= step
                    continue
                if a == b:
                    b -= step
                    continue
                
                if sigma[a] == sigma[b]:
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b
                        # Early exit for very small k
                        if abs_k <= 3:
                            break
                
                b -= step
    
    return [int(best_a), int(best_b)]
