import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s/90s kill
    
    # Known 10-term solution
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = 10
    
    # Verify known solution
    total = sum(Fraction(1, d) for d in best_sol)
    if total != 1:
        best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
        best_k = 10
    
    # Strategy: LCM-based DP with expanded candidate L values
    # For a given L (lcm of denominators), each denominator d divides L,
    # weight w = L//d, and we need subset of weights summing to L.
    
    candidate_Ls = set()
    
    # LCMs of consecutive small integers starting from 5
    for start in range(5, 12):
        for length in range(4, 12):
            l = 1
            for m in range(start, start + length):
                l = math.lcm(l, m)
                if l > 500000:
                    break
            if 5 <= l <= 500000:
                candidate_Ls.add(l)
    
    # Specific structured LCMs
    for v in [2520, 5040, 10080, 25200, 50400, 100800, 201600, 360360, 720720, 1225224, 252000, 504000]:
        if v <= 500000:
            candidate_Ls.add(v)
    
    # LCM of 5..n for various n
    for n in range(8, 16):
        l = 1
        for m in range(5, n + 1):
            l = math.lcm(l, m)
        if l <= 500000:
            candidate_Ls.add(l)
    
    # LCM of selected small integers (non-consecutive)
    small_set = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for r in range(4, 9):
        # Try combinations of r elements from first 12
        from itertools import combinations
        for combo in combinations(small_set, r):
            l = 1
            for m in combo:
                l = math.lcm(l, m)
            if l <= 500000:
                candidate_Ls.add(l)
    
    candidate_Ls = sorted(candidate_Ls)
    
    for L in candidate_Ls:
        if time.time() > deadline:
            break
        
        # Get all divisors of L that are >= 5
        divs = []
        for i in range(1, int(math.isqrt(L)) + 1):
            if L % i == 0:
                if i >= 5:
                    divs.append(i)
                other = L // i
                if other != i and other >= 5:
                    divs.append(other)
        
        if not divs:
            continue
        
        # Convert to weights
        weights = [L // d for d in divs]
        
        # DP: dp[s] = min count to reach sum s
        INF = 10**9
        dp = [INF] * (L + 1)
        dp[0] = 0
        parent = [None] * (L + 1)
        
        # Process each weight (each divisor can be used at most once)
        for idx, w in enumerate(weights):
            if w > L:
                continue
            # Reverse update for 0/1 knapsack
            for s in range(L - w, -1, -1):
                if dp[s] + 1 < dp[s + w]:
                    dp[s + w] = dp[s] + 1
                    parent[s + w] = (s, idx)
            
            if time.time() > deadline:
                break
        
        if dp[L] < INF and dp[L] < best_k:
            # Reconstruct
            sol_indices = []
            s = L
            valid = True
            while s > 0:
                if parent[s] is None:
                    valid = False
                    break
                prev_s, idx = parent[s]
                sol_indices.append(idx)
                s = prev_s
            
            if valid and len(sol_indices) == dp[L]:
                sol = [divs[i] for i in sol_indices]
                total = sum(Fraction(1, d) for d in sol)
                if total == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                    best_k = dp[L]
                    best_sol = sol
                    break
    
    # Final verification
    total = sum(Fraction(1, d) for d in best_sol)
    if total != 1 or len(set(best_sol)) != len(best_sol) or not all(5 <= d <= 10000000 for d in best_sol):
        best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
        best_k = 10
    
    return best_sol
