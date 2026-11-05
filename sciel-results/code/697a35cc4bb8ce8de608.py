"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try Greedy Algorithm first
    num, den = 1, 1
    used = set()
    result = []
    
    while num > 0:
        if time.time() > deadline:
            break
            
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
            
        d = min_d
        while d in used:
            d += 1
            
        if d > 10000000:
            break
            
        result.append(d)
        used.add(d)
        
        num = num * d - den
        den = den * d
        
        g = math.gcd(num, den)
        num //= g
        den //= g
        
    if num == 0 and len(result) > 0 and len(result) <= 256:
        return result
        
    # Fallback to DP
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
