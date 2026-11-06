import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    N = 256
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    # Precompute conflicts: for each x, which elements conflict with x
    # conflict_mask[x] is a bitset where bit y is set if x and y are in a 3-AP together
    # Two elements a, b conflict if there exists c such that {a,b,c} forms a 3-AP
    # For a fixed x, y conflicts with x if:
    #   - x is middle: y = 2x - z for some z, but we need pairs
    #   - Actually, for the B&B, we need: given current set S, which new elements can be added?
    #   - An element x can be added to S iff no element in S conflicts with x
    #   - x conflicts with y if they can form a 3-AP with some third element
    #   - x, y form a 3-AP if: (x+y) is even and (x+y)//2 is in range, or 2x-y in range, or 2y-x in range
    #   - But actually, x and y alone don't form a 3-AP. We need to check if adding x to S creates a 3-AP.
    #   - Adding x to S creates a 3-AP if:
    #     1. There exist a, b in S such that {a, b, x} is a 3-AP
    #     2. This happens if x is middle (a+b=2x), or x is endpoint (2a-x in S or 2b-x in S)
    #
    # For efficiency, let's precompute for each x the set of "forbidden" elements
    # when x is added. But this depends on S, so we can't fully precompute.
    #
    # Instead, let's use a different approach: maintain S as a bitset, and for each x,
    # precompute the "conflict pattern" - which pairs in S would conflict with x.
    #
    # Actually, the cleanest approach: for each x, precompute the set of elements y such that
    # x and y are in a 3-AP (i.e., there exists z with {x,y,z} being a 3-AP).
    # Then x can be added to S iff no element of S is in conflict_with[x].
    
    conflict_with = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x as middle: a + c = 2x, so a and c are symmetric around x
        for d in range(1, x):  # a = x-d, c = x+d
            a = x - d
            c = x + d
            if 1 <= a <= N and 1 <= c <= N:
                conflict_with[x].add(a)
                conflict_with[x].add(c)
        # x as endpoint: a + x = 2c, so c = (a+x)/2
        # Or equivalently, for each potential middle c, the other endpoint is 2c - x
        for c in range(1, N + 1):
            other = 2 * c - x
            if 1 <= other <= N and other != x:
                conflict_with[x].add(other)
    
    # Convert to bitsets for fast intersection
    import numpy as np
    conflict_bitset = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        for y in conflict_with[x]:
            mask |= (1 << y)
        conflict_bitset[x] = mask
    
    # Greedy initialization
    def greedy_construct():
        S = set()
        for x in range(1, N + 1):
            # Check if x can be added
            valid = True
            for y in S:
                if y in conflict_with[x]:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S
    
    best_set = greedy_construct()
    best_score = sum(recip[x] for x in best_set)
    
    # Branch and bound
    # We process elements in order 1, 2, ..., N
    # At each step, we decide to include or exclude
    # Pruning: if current_sum + sum of remaining weights <= best_score, prune
    
    # Precompute suffix sums for pruning
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[i]
    
    # Use iterative deepening or just DFS with pruning
    # Since N=256, we can't do full B&B in reasonable time without good pruning
    # Let's use a heuristic: process in order of decreasing weight (i.e., 1, 2, 3, ...)
    # which is already the natural order
    
    # For speed, maintain S as a Python set and use the conflict check
    # But for B&B, we need to be able to backtrack quickly
    
    # Let's use a recursive approach with memoization on (idx, frozenset(S)) 
    # but that's too large. Instead, just do DFS with pruning.
    
    # To make it faster, let's limit the depth of B&B or use a time-limited approach
    
    def bnb_solve(idx, S, current_sum):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        # Pruning
        if current_sum + suffix_sum[idx] <= best_score + 1e-12:
            return
        
        if idx == N + 1:
            if current_sum > best_score:
                best_score = current_sum
                best_set = set(S)
            return
        
        x = idx
        
        # Option 1: Include x if valid
        if x not in S:
            valid = True
            for y in S:
                if y in conflict_with[x]:
                    valid = False
                    break
            if valid:
                S.add(x)
                bnb_solve(idx + 1, S, current_sum + recip[x])
                S.discard(x)
        
        # Option 2: Exclude x
        bnb_solve(idx + 1, S, current_sum)
    
    # Run B&B starting from the greedy solution
    # Actually, let's start from empty set for B&B to find the global optimum
    # But that might be too slow. Let's try a hybrid:
    # 1. Use greedy to get a good initial solution
    # 2. Use B&B to improve, but with a time limit
    
    # Reset for B&B
    best_score_bnb = best_score
    best_set_bnb = set(best_set)
    
    # We'll use a time-limited B&B
    bnb_start = time.time()
    bnb_deadline = bnb_start + 30.0  # Give B&B 30 seconds
    
    def bnb_solve_timed(idx, S, current_sum):
        nonlocal best_score_bnb, best_set_bnb
        
        if time.time() > bnb_deadline:
            return
        
        # Pruning
        if current_sum + suffix_sum[idx] <= best_score_bnb + 1e-12:
            return
        
        if idx == N + 1:
            if current_sum > best_score_bnb:
                best_score_bnb = current_sum
                best_set_bnb = set(S)
            return
        
        x = idx
        
        # Option 1: Include x if valid
        if x not in S:
            valid = True
            for y in S:
                if y in conflict_with[x]:
                    valid = False
                    break
            if valid:
                S.add(x)
                bnb_solve_timed(idx + 1, S, current_sum + recip[x])
                S.discard(x)
        
        # Option 2: Exclude x
        bnb_solve_timed(idx + 1, S, current_sum)
    
    # Run B&B from empty set
    bnb_solve_timed(1, set(), 0.0)
    
    # Update best
    if best_score_bnb > best_score:
        best_score = best_score_bnb
        best_set = best_set_bnb
    
    # Local search polish
    def local_search(S, score):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            S_list = sorted(S, key=lambda x: recip[x])  # Low value first
            NotS_list = sorted([x for x in range(1, N + 1) if x not in S], key=lambda x: -recip[x])  # High value first
            
            found_improvement = False
            for u in S_list:
                if found_improvement or time.time() > deadline - 5.0:
                    break
                for v in NotS_list:
                    if recip[v] - recip[u] <= 0:
                        break
                    # Check if (S \ {u}) U {v} is valid
                    new_S = set(S)
                    new_S.discard(u)
                    valid = True
                    for y in new_S:
                        if y in conflict_with[v]:
                            valid = False
                            break
                    if valid:
                        new_score = score - recip[u] + recip[v]
                        if new_score > score + 1e-12:
                            S = new_S
                            S.add(v)
                            score = new_score
                            improved = True
                            found_improvement = True
                            break
            
            # Try adding
            if not found_improvement and time.time() < deadline - 5.0:
                for v in NotS_list:
                    if time.time() > deadline - 5.0:
                        break
                    valid = True
                    for y in S:
                        if y in conflict_with[v]:
                            valid = False
                            break
                    if valid:
                        S.add(v)
                        score += recip[v]
                        improved = True
                        break
        
        return S, score
    
    best_set, best_score = local_search(best_set, best_score)
    
    return sorted(best_set)
