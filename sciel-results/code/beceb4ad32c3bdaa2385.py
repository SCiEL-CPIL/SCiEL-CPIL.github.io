import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    period = 27720  # LCM(1..12)
    
    def has_matching(m, L):
        """
        Check if perfect matching exists in (m, m+L].
        Left nodes: 1..12.
        Right nodes: 0..L-1 (representing m+1 .. m+L).
        Edge (i, j) if (m+1+j) % i == 0.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            # First multiple of i strictly greater than m
            # m < k*i => k > m/i => k = floor(m/i) + 1
            k_start = (m // i) + 1
            first_val = k_start * i
            j_start = first_val - m - 1
            
            # j must be in [0, L-1]
            if j_start < 0:
                # Adjust to next multiple
                # This shouldn't happen if k_start is correct, but let's be safe
                # Actually k_start is smallest k such that k*i > m.
                # So first_val > m.
                # j_start = first_val - m - 1 >= 0.
                pass
            
            for j in range(j_start, L, i):
                adj[i - 1].append(j)
        
        # Kuhn's algorithm
        matchR = [-1] * L
        
        def dfs(u, seen):
            for v in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                if matchR[v] == -1 or dfs(matchR[v], seen):
                    matchR[v] = u
                    return True
            return False
        
        for u in range(n):
            seen = [False] * L
            if not dfs(u, seen):
                return False
        return True

    # Search for max L
    # L is at most 144
    for L in range(144, 0, -1):
        if time.time() > deadline:
            break
        found_m = -1
        for m in range(period):
            if time.time() > deadline:
                break
            # Check if f(12, m) == L
            # This requires:
            # 1. has_matching(m, L) is True
            # 2. has_matching(m, L-1) is False (if L > 1)
            
            if has_matching(m, L):
                if L == 1 or not has_matching(m, L - 1):
                    found_m = m
                    break
        
        if found_m != -1:
            return int(found_m)
            
    return 0

