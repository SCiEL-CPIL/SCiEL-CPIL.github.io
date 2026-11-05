import time
from math import gcd, isqrt

N = 512
DEADLINE = 90.0

def lcm(a, b):
    return a // gcd(a, b) * b

def is_compatible(x, y):
    return lcm(x, y) <= N

def construct():
    start_time = time.time()
    
    # Precompute compatibility
    # compat[x] is a set of y such that lcm(x,y) <= N
    # Since N=512 is small, we can do this quickly
    compat = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) <= N:
                compat[i].add(j)
                compat[j].add(i)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def check_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if not is_compatible(S_list[i], S_list[j]):
                    return False
        return True
    
    # Greedy construction
    def greedy(order):
        S = set()
        for x in order:
            if all(is_compatible(x, y) for y in S):
                S.add(x)
        return S
    
    # Local search: try to improve a set
    def local_search(S):
        S = set(S)
        improved = True
        while improved and time_left() > 5:
            improved = False
            # Try adding any missing element
            for x in range(1, N + 1):
                if x not in S and all(is_compatible(x, y) for y in S):
                    S.add(x)
                    improved = True
                    break
            if not improved:
                # Try swaps: remove y, add x
                S_list = list(S)
                for y in S_list:
                    if x is None: break # just a placeholder, logic below
                    # Actually, let's iterate properly
                    pass
                
                for y in list(S):
                    for x in range(1, N + 1):
                        if x in S: continue
                        new_S = set(S)
                        new_S.discard(y)
                        new_S.add(x)
                        if all(is_compatible(a, b) for a in new_S for b in new_S if a < b):
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
        return S
    
    # Start with the simple seed
    seed = list(range(1, isqrt(N) + 1))
    best = set(seed)
    
    # Try a few different greedy orderings
    # 1. Natural order
    S1 = greedy(range(1, N + 1))
    if len(S1) > len(best):
        best = S1
        
    # 2. Reverse natural order (larger numbers first might allow more small numbers? 
    #    Actually, large numbers have fewer compatible partners usually, so small first is better for size.
    #    But let's check.)
    S2 = greedy(range(N, 0, -1))
    if len(S2) > len(best):
        best = S2
        
    # 3. Greedy by degree (number of compatible elements in full graph)
    degrees = [len(compat[i]) for i in range(N + 1)]
    order_by_degree = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    S3 = greedy(order_by_degree)
    if len(S3) > len(best):
        best = S3
        
    # 4. Greedy by inverse degree (small numbers have high degree)
    order_by_inv_degree = sorted(range(1, N + 1), key=lambda x: degrees[x])
    S4 = greedy(order_by_inv_degree)
    if len(S4) > len(best):
        best = S4
        
    # Run local search on the best found so far
    best = local_search(best)
    
    # Try local search on other good solutions too
    for S in [S1, S2, S3, S4]:
        if time_left() < 10:
            break
        if len(S) >= len(best) - 2: # Only if it's close
            S_improved = local_search(S)
            if len(S_improved) > len(best):
                best = S_improved
    
    # Randomized restarts with local search
    import random
    while time_left() > 10:
        # Random permutation
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy(order)
        S = local_search(S)
        if len(S) > len(best):
            best = S
            
    # Final polish: try all possible single swaps on the best set
    # This is expensive O(|S| * N * |S|^2) but N is small
    if time_left() > 5:
        improved = True
        while improved and time_left() > 2:
            improved = False
            for y in list(best):
                for x in range(1, N + 1):
                    if x in best: continue
                    new_best = set(best)
                    new_best.discard(y)
                    new_best.add(x)
                    # Check validity
                    valid = True
                    for a in new_best:
                        for b in new_best:
                            if a < b and not is_compatible(a, b):
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        best = new_best
                        improved = True
                        break
                if improved:
                    break
                    
    return list(best)
