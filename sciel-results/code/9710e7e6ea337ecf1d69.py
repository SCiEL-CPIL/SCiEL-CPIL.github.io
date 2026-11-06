import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate powerful numbers up to LIMIT
    # We estimate LIMIT based on time.
    # Let's try 10^8 first. If it takes too long, we might need to reduce.
    # But 10^8 is likely fast enough for generation.
    # The bottleneck is the O(K^2) sum counting.
    
    LIMIT = 10**8
    powerful = set()
    
    # Add 1
    powerful.add(1)
    
    # Generate squares x^2
    max_x = isqrt(LIMIT)
    for x in range(1, max_x + 1):
        powerful.add(x * x)
        
    # Generate x^2 * y^3 where y is square-free and y >= 2
    # y^3 <= LIMIT => y <= (10^8)^(1/3) ~ 464
    # x^2 <= LIMIT / y^3 => x <= sqrt(LIMIT / y^3)
    for y in range(2, isqrt(LIMIT) + 1): # y^2 <= LIMIT is not the bound, y^3 <= LIMIT is
        if y * y * y > LIMIT:
            break
        # Check if y is square-free
        # y is square-free if no p^2 divides y
        is_sq_free = True
        p = 2
        temp_y = y
        while p * p <= temp_y:
            if temp_y % (p * p) == 0:
                is_sq_free = False
                break
            temp_y //= p
            p += 1 if p == 2 else 2 # 2, 3, 5...
        if not is_sq_free:
            continue
            
        max_x_y = isqrt(LIMIT // (y * y * y))
        for x in range(1, max_x_y + 1):
            # Ensure x^2 * y^3 is not already a square (i.e., y is not a square, which we ensured)
            # Actually, if y is square-free, x^2 * y^3 is powerful.
            # We just add it.
            powerful.add(x * x * y * y * y)
            
    powerful_list = sorted(powerful)
    K = len(powerful_list)
    
    # 2. Count sums
    # If K is large, K^2 might be too slow.
    # K ~ 10^4 for 10^8. K^2 ~ 10^8.
    # 10^8 operations in Python might take 30-60s.
    # We have 140s. It should be fine.
    
    if time.time() > deadline - 30:
        # Fallback to smaller limit if generation was slow
        # But generation is fast. The loop below is the risk.
        pass
        
    counts = {}
    # Optimization: use a dictionary
    for i in range(K):
        a = powerful_list[i]
        if a * 2 > LIMIT * 2: # Max sum is 2*LIMIT
            break
        for j in range(i, K):
            b = powerful_list[j]
            s = a + b
            if s > LIMIT * 2:
                break
            counts[s] = counts.get(s, 0) + 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # 3. Verify and extract pairs
    # The count from the loop is exact for sums of elements in powerful_list.
    # Since powerful_list contains all powerful numbers <= LIMIT,
    # and best_N <= 2*LIMIT, all pairs (a,b) with a+b=best_N must have a,b <= LIMIT.
    # So the count is correct.
    
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
