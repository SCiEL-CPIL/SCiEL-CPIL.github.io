import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    
    # Compute largest prime factor for all n up to target
    P = [0] * (target + 1)
    P[1] = 1
    
    for j in range(2, target + 1, 2):
        P[j] = 2
    
    for p in range(3, target + 1, 2):
        if P[p] == 0:  # p is prime
            for j in range(p, target + 1, p):
                P[j] = p
    
    # Binary search on threshold K
    # For a given K, check if there exists a in [1, target//2] such that
    # P[a] <= K and P[target-a] <= K
    
    def exists_pair(K):
        """Check if there's a pair (a, b) with a+b=target and max(P[a],P[b]) <= K"""
        for a in range(1, target // 2 + 1):
            if P[a] <= K and P[target - a] <= K:
                return a
        return None
    
    # Find the minimum K such that a valid pair exists
    # Lower bound: at least 2 (since target > 2)
    # Upper bound: max(P[a]) for any a, which is at most target
    lo = 2
    hi = target
    
    best_a = 1
    best_b = target - 1
    best_K = P[target - 1]  # fallback
    
    while lo <= hi:
        mid = (lo + hi) // 2
        a_found = exists_pair(mid)
        
        if a_found is not None:
            best_a = a_found
            best_b = target - a_found
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
        
        if time.time() > deadline:
            break
    
    # If binary search didn't converge, do a linear scan as fallback
    if best_K == P[target - 1] and best_a == 1:
        min_max_p = float('inf')
        for a in range(1, target // 2 + 1):
            b = target - a
            pa = P[a]
            pb = P[b]
            current_max = pa if pa > pb else pb
            if current_max < min_max_p:
                min_max_p = current_max
                best_a = a
                best_b = b
                if min_max_p <= 2:
                    break
            if time.time() > deadline:
                break
    
    return [best_a, best_b]
