import time
import math

def construct():
    S = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill

    # Step 1: Compute largest prime factor for all numbers up to S
    # lpf[n] = largest prime factor of n, lpf[1] = 1
    lpf = [0] * (S + 1)
    lpf[1] = 1
    for i in range(2, S + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, S + 1, i):
                lpf[j] = i
    
    if time.time() > deadline:
        return [1, S - 1]

    # Step 2: Bucket numbers by their lpf value
    # numbers_by_lpf[p] = list of numbers x in [1, S] with lpf[x] == p
    numbers_by_lpf = {}
    for x in range(1, S + 1):
        p = lpf[x]
        if p not in numbers_by_lpf:
            numbers_by_lpf[p] = []
        numbers_by_lpf[p].append(x)
    
    if time.time() > deadline:
        return [1, S - 1]

    # Step 3: Binary search on the smoothness bound B
    # For a given B, generate all B-smooth numbers (lpf[x] <= B) up to S,
    # then check if any pair sums to S using a hash set.
    
    distinct_lpfs = sorted(numbers_by_lpf.keys())
    
    # Helper: check if there exists a pair (a, b) with a+b=S and both B-smooth
    def check_smooth_pair(B):
        smooth_set = set()
        for p in distinct_lpfs:
            if p > B:
                break
            for x in numbers_by_lpf[p]:
                smooth_set.add(x)
        
        for x in smooth_set:
            comp = S - x
            if comp in smooth_set:
                return [x, comp]
        return None
    
    # Binary search for minimal B
    lo = 1
    hi = distinct_lpfs[-1]  # maximum lpf value
    best_result = None
    
    while lo <= hi:
        if time.time() > deadline:
            if best_result:
                return best_result
            return [1, S - 1]
        
        mid = (lo + hi) // 2
        result = check_smooth_pair(mid)
        
        if result is not None:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_result:
        return best_result
    return [1, S - 1]
