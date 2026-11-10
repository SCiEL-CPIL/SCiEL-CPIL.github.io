import numpy as np
import math
import time
from collections import defaultdict
import bisect

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    N = 10_000_000
    
    # Phase 1: Compute sigma(n) for all n using vectorized additive sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 100000 == 0 and time.time() > deadline - 30:
            N = i
            break
    
    if time.time() > deadline - 20:
        return [11, 6]  # Fallback
    
    # Phase 2: Group numbers by sigma value
    # Use argsort approach for efficiency
    sigma_vals = sigma[1:N+1]
    sorted_indices = np.argsort(sigma_vals, kind='stable')
    sorted_sigmas = sigma_vals[sorted_indices]
    
    # Phase 3: Search for best pair in each fiber
    # Error = |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this
    best_a, best_b = 11, 6
    # Initial error: |2*11 - 3*6| / (2*6) = |22-18|/12 = 4/12 = 1/3
    best_num = 4   # numerator of error
    best_den = 12  # denominator of error
    
    # Process groups using two-pointer technique on sorted fiber
    i = 0
    while i < N:
        if time.time() > deadline - 10:
            break
        s = sorted_sigmas[i]
        j = i
        while j < N and sorted_sigmas[j] == s:
            j += 1
        
        if j - i >= 2:
            # Get the numbers in this fiber, sorted
            nums = sorted_indices[i:j] + 1  # +1 because we sliced from index 1
            nums_sorted = np.sort(nums)
            nlen = len(nums_sorted)
            
            # Two-pointer approach: for each b, find a closest to 3b/2
            # Since both a and b come from the same sorted list,
            # we can use binary search for each b
            for idx_b in range(nlen):
                if time.time() > deadline - 5:
                    break
                
                b = int(nums_sorted[idx_b])
                target_a = 3 * b // 2
                
                # Binary search for closest to target_a
                pos = bisect.bisect_left(nums_sorted, target_a)
                
                # Check pos and pos-1
                for idx_a in (pos, pos - 1):
                    if idx_a < 0 or idx_a >= nlen:
                        continue
                    a = int(nums_sorted[idx_a])
                    if a == b:
                        continue
                    
                    # Compute error: |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    
                    # Compare num/den vs best_num/best_den
                    # num * best_den < best_num * den
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b
        
        i = j
    
    # Phase 4: Targeted search using ratio constraint 2a - 3b = k
    # For small |k|, the error |a/b - 3/2| = |k|/(2b) is small when b is large.
    # We search for pairs (a, b) with a <= 10^7, b <= 10^7, sigma(a) = sigma(b),
    # and 2a - 3b = k for small k values.
    if time.time() < deadline - 5:
        for k in range(-50, 51):
            if time.time() > deadline - 5:
                break
            # a = (3b + k) / 2 must be integer, so 3b + k must be even
            # 3b + k even => b + k even (since 3b has same parity as b)
            # So b must have same parity as k
            min_b = max(1, (2 - k + 1) // 3)  # ceil((2-k)/3)
            max_b = min(N, (2 * N - k) // 3)
            
            # Adjust min_b for parity
            if k % 2 != 0:
                # b must be odd
                if min_b % 2 == 0:
                    min_b += 1
                step = 2
            else:
                # b must be even
                if min_b % 2 == 0:
                    pass
                else:
                    min_b += 1
                step = 2
            
            # For each b in range, compute a and check sigma
            # Limit the number of checks to avoid timeout
            checks = 0
            for b in range(min_b, max_b + 1, step):
                if time.time() > deadline - 5:
                    break
                if checks > 100000:
                    break
                checks += 1
                a = (3 * b + k) // 2
                if a < 1 or a > N:
                    continue
                if a == b:
                    continue
                # Check if both are within our computed sigma range
                if a <= N and b <= N:
                    if sigma[a] == sigma[b]:
                        num = abs(2*a - 3*b)
                        den = 2*b
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
