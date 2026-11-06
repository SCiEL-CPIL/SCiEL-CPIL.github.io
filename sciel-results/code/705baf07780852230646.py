import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We will generate powerful numbers up to 10^14
    # This allows for ~10^7 numbers, which is manageable.
    LIMIT = 10**14
    
    # 1. Generate squarefree numbers up to (LIMIT)^(1/3)
    max_b = int(LIMIT ** (1/3)) + 2
    is_squarefree = [True] * (max_b + 1)
    is_squarefree[0] = False
    is_squarefree[1] = True
    
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_squarefree[j] = False
            
    # 2. Generate powerful numbers
    powerful_set = set()
    powerful_set.add(1) # 1 is powerful
    
    for b in range(1, max_b + 1):
        if not is_squarefree[b]:
            continue
        b3 = b ** 3
        if b3 > LIMIT:
            break
        max_a = isqrt(LIMIT // b3)
        # Iterate a
        # Optimization: if max_a is large, this loop is heavy.
        # For b=1, max_a = 10^7.
        # Total ops ~ 1.5 * 10^7.
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= LIMIT:
                powerful_set.add(val)
            else:
                break
                
    powerful_list = sorted(powerful_set)
    
    # Check time
    if time.time() > deadline - 30:
        # If generation took too long, we might not have time to search.
        # But 10^7 ops should be < 2s.
        pass

    # 3. Find best N
    # We will test candidate N values.
    # Since r(N) behavior is unknown, we sample a wide range.
    # Focus on the upper end of our generated range, as larger N might have more representations
    # due to the sheer number of available powerful numbers, even if density drops.
    
    candidates = set()
    # Add some round numbers
    for exp in range(8, 15):
        base = 10 ** exp
        for k in [1, 2, 3, 5, 7, 10]:
            n = k * base
            if n <= LIMIT:
                candidates.add(n)
    
    # Add values near the limit
    for i in range(0, 100, 10):
        n = LIMIT - i * 10**9
        if n > 0:
            candidates.add(n)
            
    # Add some random-ish values
    import random
    random.seed(42)
    for _ in range(200):
        n = random.randint(10**8, LIMIT)
        candidates.add(n)
        
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Sort candidates to check larger ones first? Or smaller?
    # Let's just iterate.
    for N in candidates:
        if time.time() > deadline - 5:
            break
            
        count = 0
        pairs = []
        # Iterate through powerful numbers a <= N/2
        # Using binary search to find the end of the list to iterate
        # Or just iterate and break
        for a in powerful_list:
            if 2 * a > N:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
                
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
            
    if best_N is None:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}
