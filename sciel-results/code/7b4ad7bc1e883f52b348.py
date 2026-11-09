import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [11, 6]

    best_err = 1.0 / 3.0  # From (11, 6)
    best_pair = (11, 6)
    
    # Iterate b from N down to 1
    for b in range(N, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Check break condition
        if 1.0 / (2.0 * b) > best_err:
            break
            
        target_sigma = sigma[b]
        
        # Determine step for k
        if b % 2 == 0:
            start_k = 2
            step_k = 2
        else:
            start_k = 1
            step_k = 2
            
        # Max k such that k/(2*b) < best_err
        max_k = int(2 * b * best_err)
        
        if max_k < start_k:
            continue
            
        for k in range(start_k, max_k + 1, step_k):
            # Check a1 = (3b + k) / 2
            num1 = 3 * b + k
            if num1 % 2 == 0:
                a1 = num1 // 2
                if 1 <= a1 <= N:
                    if sigma[a1] == target_sigma:
                        new_err = k / (2.0 * b)
                        if new_err < best_err:
                            best_err = new_err
                            best_pair = (a1, b)
                        break
                        
            # Check a2 = (3b - k) / 2
            num2 = 3 * b - k
            if num2 > 0 and num2 % 2 == 0:
                a2 = num2 // 2
                if 1 <= a2 <= N:
                    if sigma[a2] == target_sigma:
                        new_err = k / (2.0 * b)
                        if new_err < best_err:
                            best_err = new_err
                            best_pair = (a2, b)
                        break
                        
    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
