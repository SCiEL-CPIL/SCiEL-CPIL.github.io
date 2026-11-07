"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(n, m) is the
    least positive L such that (m, m+L] contains distinct b_1,...,b_n with
    i dividing b_i.
    
    We use a bipartite matching approach to check feasibility for
    a given L, and exhaustive search over one period to find the worst-case start.
    """
    n = 12
    max_m = 1000000
    max_L = 144  # Upper bound from task description
    
    # LCM of 1..12 is 27720
    period = 27720
    
    def feasible(m, L):
        """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
        # Values are m+1 to m+L
        # We need to match indices 1..n to distinct values in (m, m+L]
        
        # match[j] stores the index i (1-based) that is matched to value m+1+j
        # j is 0-indexed offset in (m, m+L]
        match = [-1] * L
        
        # For each index i, we need to find an augmenting path
        # We use a simple DFS based bipartite matching
        
        # To speed up, we precompute for each i which offsets j are valid
        # valid_offsets[i] = list of j such that (m + 1 + j) % i == 0
        
        # Actually, we can just iterate j in 0..L-1 and check divisibility on the fly
        # or precompute for the specific m.
        
        # Let's precompute adjacency for this specific m, L
        # adj[i] = list of j (0-indexed) such that (m+1+j) % i == 0
        adj = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(L):
                if (m + 1 + j) % i == 0:
                    adj[i].append(j)
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match[j] == -1 or dfs(match[j], seen):
                    match[j] = i
                    return True
            return False
        
        count = 0
        for i in range(1, n + 1):
            seen = [False] * L
            if dfs(i, seen):
                count += 1
        
        return count == n

    best_m = 0
    best_L = 0
    
    # Search over one period
    # f(12, m) is periodic with period 27720
    for m in range(period):
        # Binary search for smallest L
        lo, hi = 1, max_L
        result_L = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(m, mid):
                result_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if result_L > best_L:
            best_L = result_L
            best_m = m
    
    # The optimal m in [0, 1000000] will be one of the m in [0, period-1]
    # We return the smallest such m to be deterministic, or any.
    # The task asks for ONE int m.
    return int(best_m)

# EVOLVE-BLOCK-END
