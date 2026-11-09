import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Target limit for powerful numbers
    # 10^15 generates ~32k numbers, ~500M pair checks.
    # 10^14 generates ~15k numbers, ~100M pair checks.
    # Let's try 10^15 and break if too slow.
    LIMIT = 10**15
    
    powerful = set()
    
    # Generate powerful numbers up to LIMIT
    max_v = int(LIMIT ** (1/3))
    for v in range(1, max_v + 1):
        v3 = v ** 3
        max_u = isqrt(LIMIT // v3)
        for u in range(1, max_u + 1):
            p = (u * u) * v3
            if p <= LIMIT:
                powerful.add(p)
                
    pw = sorted(powerful)
    pw_set = set(pw)
    n = len(pw)
    
    counts = defaultdict(int)
    
    # If the list is too large, we might not finish.
    # n ~ 32000. n^2/2 ~ 5e8.
    # In Python, 5e8 iterations is ~30-60s.
    # We will monitor time.
    
    for i in range(n):
        a = pw[i]
        
        # Check time every 1000 outer iterations
        if i % 1000 == 0:
            if time.time() > deadline - 5.0:
                # Return best found so far
                if not counts:
                    return {'N': 2, 'pairs': [[1, 1]]}
                best_N = max(counts, key=counts.get)
                pairs = []
                for x in pw:
                    y = best_N - x
                    if y < x: continue
                    if y in pw_set:
                        pairs.append([x, y])
                return {'N': best_N, 'pairs': pairs}

        for j in range(i, n):
            b = pw[j]
            s = a + b
            counts[s] += 1
            
    # Find best
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    pairs = []
    for x in pw:
        y = best_N - x
        if y < x: continue
        if y in pw_set:
            pairs.append([x, y])
            
    return {'N': best_N, 'pairs': pairs}
