import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 128
    # Start with the upper half, which is a valid set
    # Sums of any two distinct elements >= 65+66 = 131 > 128, so no sums in set.
    initial_S = set(range(N//2 + 1, N + 1))
    
    best_S = set(initial_S)
    best_size = len(best_S)
    
    # Helper to check if adding x to current set S is valid
    def is_valid_add(x, S):
        # Check all pairs u, v in S (u != v)
        # Forbidden if u+v, u+x, v+x are all in S U {x}
        # Since x not in S, u+x and v+x cannot be x (u,v >= 1).
        # So u+x and v+x must be in S.
        # u+v must be in S or equal to x.
        
        # Optimization: Precompute u+x and v+x? No, x is fixed.
        # Just loop.
        for u in S:
            ux = u + x
            if ux not in S:
                continue
            for v in S:
                if u == v:
                    continue
                vx = v + x
                if vx not in S:
                    continue
                # If we reach here, u+x and v+x are in S.
                # Check u+v
                uv = u + v
                if uv == x or uv in S:
                    return False
        return True

    # Strategy 1: Greedy descending
    S = set(initial_S)
    for x in range(N//2, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_add(x, S):
            S.add(x)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
        
    # Strategy 2: Randomized restarts
    # We try to find a better ordering of adding numbers 1..N//2
    # We stop if we run out of time.
    
    # Set a seed for reproducibility? No, random is fine for search.
    # But for stable evaluation, maybe fix seed? The task doesn't require determinism, 
    # but "evaluator measures raw value". If random gives different results, 
    # it's risky if it sometimes gets worse. 
    # However, we keep best_S, so it can only improve from initial 64.
    # The greedy descending already improves to ~66.
    # Random restarts might find 67+.
    
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        S = set(initial_S)
        candidates = list(range(1, N//2 + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
            
        # Early termination if we hit a theoretical upper bound? 
        # Hard to know the bound. 
        # If we find a very large set, we can stop.
        # But let's just run until time limit.
        
    return list(best_S)
