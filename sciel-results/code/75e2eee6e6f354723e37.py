import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    LIMIT = 512
    candidates = list(range(1, LIMIT + 1))
    
    # Precompute lcm compatibility: compat[a][b] = True if lcm(a,b) <= LIMIT
    # Use a set-based approach for speed
    # Actually, let's precompute a compatibility matrix as a list of sets
    # For each x, store the set of y such that lcm(x,y) <= LIMIT
    compat = [set() for _ in range(LIMIT + 1)]
    for a in range(1, LIMIT + 1):
        for b in range(a, LIMIT + 1):
            g = gcd(a, b)
            if a * b // g <= LIMIT:
                compat[a].add(b)
                compat[b].add(a)
    
    def is_compatible_with_set(x, S_set):
        """Check if x is compatible with all elements in S_set."""
        for y in S_set:
            if y not in compat[x]:
                return False
        return True
    
    def greedy_add(S_set, remaining):
        """Greedily add compatible elements from remaining to S_set."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in list(remaining):
                if is_compatible_with_set(x, S_set):
                    S_set.add(x)
                    remaining.discard(x)
                    changed = True
                    break
        return S_set
    
    def local_search(S_set, remaining, max_iter=1000):
        """Try 1-for-1, 1-for-2, 2-for-1 swaps."""
        best_S = set(S_set)
        best_size = len(best_S)
        remaining = set(remaining)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # 1-for-1: remove x from S, add y from remaining
            for x in list(best_S):
                if time.time() > deadline - 5:
                    break
                S_without_x = best_S - {x}
                for y in list(remaining):
                    if y in best_S:
                        continue
                    if is_compatible_with_set(y, S_without_x):
                        new_S = S_without_x | {y}
                        if len(new_S) > best_size:
                            best_S = new_S
                            best_size = len(best_S)
                            remaining = set(candidates) - best_S
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # 1-for-2: remove x from S, add y1, y2 from remaining
                found = False
                for x in list(best_S):
                    if time.time() > deadline - 5:
                        break
                    S_without_x = best_S - {x}
                    # Find pairs y1,y2 from remaining compatible with S_without_x and each other
                    compatible_with_S = [y for y in remaining if is_compatible_with_set(y, S_without_x)]
                    for i in range(len(compatible_with_S)):
                        for j in range(i+1, len(compatible_with_S)):
                            y1 = compatible_with_S[i]
                            y2 = compatible_with_S[j]
                            if y2 in compat[y1]:
                                new_S = S_without_x | {y1, y2}
                                if len(new_S) > best_size:
                                    best_S = new_S
                                    best_size = len(best_S)
                                    remaining = set(candidates) - best_S
                                    found = True
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                
                # 2-for-1: remove x1,x2 from S, add y from remaining
                found = False
                S_list = list(best_S)
                for i in range(len(S_list)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i+1, len(S_list)):
                        x1, x2 = S_list[i], S_list[j]
                        S_without = best_S - {x1, x2}
                        for y in remaining:
                            if y in best_S:
                                continue
                            if is_compatible_with_set(y, S_without):
                                new_S = S_without | {y}
                                if len(new_S) > best_size:
                                    best_S = new_S
                                    best_size = len(best_S)
                                    remaining = set(candidates) - best_S
                                    found = True
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                
                # No improvement, break
                break
        
        return best_S
    
    # Multiple restarts with different strategies
    best_result = set()
    best_size = 0
    
    # Strategy 1: Greedy from scratch
    S1 = set([1])  # Always include 1
    remaining1 = set(candidates) - S1
    S1 = greedy_add(S1, remaining1)
    S1 = local_search(S1, set(candidates) - S1, max_iter=500)
    if len(S1) > best_size:
        best_size = len(S1)
        best_result = S1
    
    if time.time() > deadline - 10:
        return list(best_result)
    
    # Strategy 2: Start with powers of 2
    powers_of_2 = set()
    p = 1
    while p <= LIMIT:
        powers_of_2.add(p)
        p *= 2
    S2 = set(powers_of_2)
    remaining2 = set(candidates) - S2
    S2 = greedy_add(S2, remaining2)
    S2 = local_search(S2, set(candidates) - S2, max_iter=500)
    if len(S2) > best_size:
        best_size = len(S2)
        best_result = S2
    
    if time.time() > deadline - 10:
        return list(best_result)
    
    # Strategy 3: Randomized restarts
    import random
    rng = random.Random(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random initial set
        S_r = set()
        # Include 1
        S_r.add(1)
        # Add some random small numbers
        for x in range(2, min(30, LIMIT+1)):
            if rng.random() < 0.3:
                if is_compatible_with_set(x, S_r):
                    S_r.add(x)
        
        remaining_r = set(candidates) - S_r
        S_r = greedy_add(S_r, remaining_r)
        S_r = local_search(S_r, set(candidates) - S_r, max_iter=300)
        
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_result = S_r
    
    # Final: try to improve best_result with more local search
    if time.time() < deadline - 5:
        remaining_final = set(candidates) - best_result
        best_result = local_search(best_result, remaining_final, max_iter=1000)
    
    # Ensure nonempty
    if not best_result:
        best_result = {1}
    
    return list(best_result)
