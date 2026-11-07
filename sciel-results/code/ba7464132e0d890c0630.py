import sys
import time
import math

# EVOLVE-BLOCK-START
import numpy as np

MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # 1. Compute sigma(n) for all n <= MAX_N using NumPy for speed
    # sigma[n] = sum of all positive divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # For each divisor d, add d to all multiples of d
    # This is O(N log N) but NumPy vectorization makes it fast
    for d in range(1, MAX_N + 1):
        # Add d to sigma[d], sigma[2d], sigma[3d], ...
        sigma[d::d] += d
        
        # Periodic time check
        if d % 100000 == 0 and time.time() > deadline:
            break
    
    if time.time() > deadline:
        return [11, 6]
    
    # 2. Search for best pair (a, b)
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = |2a - 3b|. Want small k and large b.
    
    best_a, best_b = 11, 6
    best_err_num, best_err_den = 4, 12  # Error for 11/6 is 1/3
    best_score_val = 1.0 / 3.0
    
    # Convert to list for faster indexing in the search loop
    sig_list = sigma.tolist()
    
    # Iterate b from MAX_N down to 1
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline:
            break
            
        # Pruning: minimum possible error for this b is 1/(2*b)
        # If that's worse than best, we can stop
        if 1.0 / (2.0 * b) >= best_score_val:
            break
            
        sb = sig_list[b]
        base = 3 * b // 2  # floor(3b/2)
        
        # Check candidates a near 1.5*b
        # k = |2a - 3b|
        # If 3b is odd: base = (3b-1)//2, so 2*base - 3b = -1, k=1
        #                base+1 gives k=1
        # If 3b is even: base = 3b//2, so 2*base - 3b = 0, k=0 (invalid)
        #                base±1 gives k=2
        
        three_b = 3 * b
        found_improvement = False
        
        if three_b % 2 == 1:
            # 3b is odd, min k = 1
            # Check a = base and a = base + 1 (both give k=1)
            for a in [base, base + 1]:
                if 1 <= a <= MAX_N:
                    if sig_list[a] == sb:
                        # k = 1, error = 1/(2b)
                        if 1 * best_err_den < best_err_num * 2 * b:
                            best_a, best_b = a, b
                            best_err_num, best_err_den = 1, 2 * b
                            best_score_val = 1.0 / (2.0 * b)
                            found_improvement = True
                            break  # k=1 is best possible for this b
        else:
            # 3b is even, min non-zero k = 2
            # Check a = base ± 1 (k=2), then base ± 2 (k=4), etc.
            for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                a = base + offset
                if a < 1 or a > MAX_N:
                    continue
                if sig_list[a] == sb:
                    k = abs(2 * a - three_b)
                    if k == 0:
                        continue
                    # Compare k/(2b) with best_err_num/best_err_den
                    if k * best_err_den < best_err_num * 2 * b:
                        best_a, best_b = a, b
                        best_err_num, best_err_den = k, 2 * b
                        best_score_val = k / (2.0 * b)
                        found_improvement = True
                        if k == 2:
                            break  # k=2 is best possible for even 3b
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
