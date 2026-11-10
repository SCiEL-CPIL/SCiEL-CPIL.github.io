"""Worst interval starts for distinct-multiple matching."""
import time
from collections import deque

# EVOLVE-BLOCK-START

def _compute_f(m, n=12):
    """Compute f(n, m): least L>0 s.t. (m, m+L] has distinct b_i with i|b_i."""
    # For each i in 1..n, find the smallest multiple of i in (m, m+L]
    # We need L such that there exist distinct b_i with i|b_i and m < b_i <= m+L
    # b_i must be distinct
    
    # For a given L, check if matching exists
    def can_match(L):
        # For each i, the set of valid b values: multiples of i in (m, m+L]
        # b ranges from m+1 to m+L
        # We need to assign distinct b to each i such that i|b
        # Build adjacency: i -> list of b values in (m, m+L] divisible by i
        
        # Use bitset matching via DFS (n=12 is small)
        # For each i, candidates are multiples of i in [m+1, m+L]
        
        # Build list of candidates for each i
        # To keep it fast, represent as set of b-values
        # n=12, L<=144, so at most 144 b-values
        
        # Use standard bipartite matching
        # Left side: i=1..n (12 nodes)
        # Right side: b values in (m, m+L] (at most L values, L<=144)
        
        # For efficiency, map b to index
        # b ranges from m+1 to m+L
        # For each i, valid b's are those where b % i == 0
        
        # Build adjacency list
        # adj[i] = set of b indices that work for i
        
        # Actually, let's use a simpler approach:
        # For each i, the smallest multiple of i > m is: ceil((m+1)/i) * i
        # Then next multiples: +i, +2i, ...
        # All must be <= m+L
        
        # For matching, we need to assign distinct b's
        # Use DFS-based bipartite matching (Kuhn's algorithm)
        
        # Map b-value to 0-indexed position
        # b = m+1, m+2, ..., m+L
        # index = b - (m+1), range [0, L-1]
        
        # adj[i] = list of indices j where (m+1+j) % i == 0
        adj = [[] for _ in range(n)]
        for i in range(1, n+1):
            # multiples of i in (m, m+L]
            # first multiple > m: 
            # q = (m // i) + 1, first = q * i
            # need first <= m + L
            q = (m // i) + 1
            first = q * i
            if first <= m + L:
                # indices: (first - m - 1), (first - m - 1) + i, ... while <= L-1
                idx = first - m - 1
                while idx < L:
                    adj[i-1].append(idx)
                    idx += i
        
        # Now run bipartite matching
        # match[j] = which i is matched to b-index j, or -1
        match = [-1] * L
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match[j] == -1 or dfs(match[j], visited):
                    match[j] = i
                    return True
            return False
        
        found = 0
        for i in range(n):
            visited = [False] * L
            if dfs(i, visited):
                found += 1
        
        return found == n
    
    # Binary search on L
    # Lower bound: for each i, the smallest b_i > m divisible by i is ceil((m+1)/i)*i
    # The max of these gives a lower bound on L
    # Actually L must be at least max_i (ceil((m+1)/i)*i - m)
    # But also need distinctness, so L could be larger
    
    # Upper bound: 144 (given in problem)
    # Let's find a tight lower bound
    lo = 1
    hi = 144
    
    # Quick lower bound
    for i in range(1, n+1):
        q = (m // i) + 1
        first = q * i
        lb = first - m
        if lb > lo:
            lo = lb
    
    if lo > hi:
        return hi  # shouldn't happen
    
    while lo < hi:
        mid = (lo + hi) // 2
        if can_match(mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo


def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 50  # 50s safety margin (90s limit)
    
    best_m = 0
    best_f = 0
    
    # The function f(12, m) is periodic with period LCM(1..12) = 27720.
    # f(12, m) depends only on m % 27720.
    # We can exhaustively search m in [0, 27719].
    # This is 27,720 evaluations.
    # Each evaluation involves a binary search on L (1..144) and a bipartite match.
    # 27720 * 8 * (small match cost) is very fast.
    
    period = 27720
    
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    # If we have time, we can verify that no other m in [0, 1000000] is better,
    # but periodicity guarantees m % period is the only factor.
    # So best_m in [0, period) is a valid answer in [0, 1000000].
    # We return the smallest such m.
    
    return int(best_m)

# EVOLVE-BLOCK-END
