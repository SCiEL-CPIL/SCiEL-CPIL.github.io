import time
from math import gcd

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct b_1,...,b_12 with i | b_i.
    """
    N = 12
    MAX_M = 1000000
    # LCM of 1..12
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = 1
    for i in range(1, N + 1):
        M = lcm(M, i)
        
    # M is 27720
    # f(12, m) depends only on m % M
    # We want to find r in [0, M-1] that maximizes f(12, r)
    # Then we can pick any m in [0, MAX_M] such that m % M == r.
    # Since M = 27720 < 1000000, such an m always exists (e.g., m=r).
    
    def has_matching(r, L):
        """
        Check if (r, r+L] contains distinct b_1..b_12 with i | b_i.
        r is the effective start modulo M, but the interval is (r, r+L].
        Note: The problem defines the interval as (m, m+L].
        So candidates are integers x such that r < x <= r+L.
        """
        # Collect all integers in (r, r+L]
        # There are exactly L such integers.
        # We need to assign them to 1..12 such that i divides assigned_x.
        
        # Bipartite matching
        # Left: 1..12
        # Right: 0..L-1 representing the integers r+1, ..., r+L
        
        # adj[i] = list of right-node indices j such that (r+j) is divisible by i
        adj = [[] for _ in range(N + 1)]
        
        for j in range(1, L + 1):
            x = r + j
            for i in range(1, N + 1):
                if x % i == 0:
                    adj[i].append(j)
        
        # Kuhn's algorithm for bipartite matching
        # match_right[j] = i  if right node j is matched to left node i
        # We want to match all left nodes 1..12
        
        match_right = [-1] * (L + 1) # 1-indexed for right nodes
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_right[j] == -1 or dfs(match_right[j], visited):
                    match_right[j] = i
                    return True
            return False
        
        for i in range(1, N + 1):
            visited = [False] * (L + 1)
            if not dfs(i, visited):
                return False
        return True

    def compute_f(r):
        # Find least L in 1..144 such that has_matching(r, L) is True
        for L in range(1, 145):
            if has_matching(r, L):
                return L
        return 144 # Should not happen, but safe

    best_r = 0
    best_f = 0
    
    # Exhaustive search over one period
    # 27720 iterations, each doing up to 144 matching checks.
    # Matching check is very fast (small graph).
    
    for r in range(M):
        f_val = compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
            
    # We found best_r. We need to return an m in [0, MAX_M] such that m % M == best_r.
    # m = best_r is a valid candidate since best_r < M < MAX_M.
    # We can also pick other values, but best_r is simple.
    
    return int(best_r)

# EVOLVE-BLOCK-END
