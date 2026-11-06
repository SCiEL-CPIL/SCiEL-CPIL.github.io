import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate powerful numbers up to 10^12
    # x^2 * y^3 <= 10^12
    # y <= (10^12)^(1/3) = 10000
    limit_val = 10**12
    powerful = set()
    for y in range(1, 10001):
        y3 = y ** 3
        if y3 > limit_val:
            break
        max_x = isqrt(limit_val // y3)
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= limit_val:
                powerful.add(val)
    
    # Sort for potential early termination or just list
    p_list = sorted(powerful)
    
    # Count sums
    # If len(p_list) is ~10000, pairs are 50 million.
    # This is feasible.
    counts = {}
    for i, a in enumerate(p_list):
        if time.time() > deadline:
            break
        for b in p_list[i:]:
            if a + b > limit_val * 2: # Not a hard limit for N, but N can be up to 10^18
                # Actually N can be up to 10^18. 
                # But if a,b <= 10^12, sum <= 2*10^12.
                # We are only finding N within this range.
                pass
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            
    # Find max
    best_N = 0
    max_count = 0
    for n, c in counts.items():
        if c > max_count:
            max_count = c
            best_N = n
            
    # Retrieve pairs
    pairs = []
    if best_N > 0:
        p_set = powerful
        for a in p_list:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in p_set:
                pairs.append([a, b])
                
    return {'N': best_N, 'pairs': pairs}
