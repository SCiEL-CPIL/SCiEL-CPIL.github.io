import time
import math

def compute_score(U):
    if not U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sum_set = set()
    diff_set = set()
    
    U_list = list(U_set)
    n = len(U_list)
    
    for i in range(n):
        for j in range(n):
            sum_set.add(U_list[i] + U_list[j])
            diff_set.add(U_list[i] - U_list[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def generate_block_set(S, M, K):
    U = set()
    for i in range(K):
        base = i * M
        for s in S:
            U.add(base + s)
    return list(U)

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Try different block structures
    # S is the set of offsets, M is the gap
    # We want S to be dense to maximize collisions in sums?
    # Actually, for the ratio to be high, we want |U-U| large and |U+U| small.
    # My analysis: Ratio ~ K. Score ~ 2.
    # Let's try S = {0, 1, ..., d-1}, M = 2d - 1.
    
    candidates = []
    
    # d=1: S={0}, M=1. U={0, 1, ..., K-1}. Ratio 1. Score 1.
    # d=2: S={0, 1}, M=3.
    # d=3: S={0, 1, 2}, M=5.
    # d=4: S={0, 1, 2, 3}, M=7.
    
    for d in range(2, 10):
        S = list(range(d))
        M = 2 * d - 1
        
        # Estimate max K
        # Time limit for scoring is the bottleneck.
        # N = K * d.
        # Ops ~ N^2.
        # Let's try K such that N is around 1000-2000.
        K_est = 1000 // d
        if K_est < 10: K_est = 10
        
        # We can try a few K values
        for K in [K_est, K_est * 2, K_est * 4]:
            if time.time() > deadline - 5:
                break
            
            U = generate_block_set(S, M, K)
            score = compute_score(U)
            
            if score > best_score:
                best_score = score
                best_set = U
                
            # Check time
            if time.time() > deadline - 10:
                break
                
    return best_set
