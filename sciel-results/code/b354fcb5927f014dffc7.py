import time
import sys
from collections import defaultdict
import bisect

def construct():
    # Deadline guard: 85 seconds max (evaluator kills at 90s)
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using O(N log log N) sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # 2. Group numbers by their sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        sigma_groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    max_b = int(N * 2 / 3)
    
    # Iterate b from large to small
    for b in range(max_b, 0, -1):
        # Pruning: minimum possible error for this b is 1/(2b)
        if 1.0 / (2.0 * b) >= best_error:
            break
            
        sb = sigma[b]
        target_a = 1.5 * b
        
        # Get all a's with same sigma as b
        group = sigma_groups[sb]
        if not group:
            continue
        
        # Binary search for closest values to target_a
        # Find insertion point
        lo, hi = 0, len(group)
        while lo < hi:
            mid = (lo + hi) // 2
            if group[mid] < target_a:
                lo = mid + 1
            else:
                hi = mid
        # lo is the first index where group[lo] >= target_a
        # Check lo and lo-1 (and a few neighbors for safety)
        for idx in range(max(0, lo - 3), min(len(group), lo + 3)):
            a = group[idx]
            if a == b:
                continue
            err_num = abs(2 * a - 3 * b)
            if err_num == 0:
                continue
            err_den = 2 * b
            error = err_num / err_den
            
            if error < best_error:
                best_error = error
                best_a = a
                best_b = b
                
        # Time check
        if time.time() > deadline - 3:
            break

    # Ensure types are Python ints
    return [int(best_a), int(best_b)]

# Test / Sanity Check if run directly
if __name__ == "__main__":
    res = construct()
    print(res)
    a, b = res
    if isinstance(a, int) and isinstance(b, int):
        print(f"a={a}, b={b}")
    else:
        print("Type error")
