import time
from math import gcd

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct b_1,...,b_12 with i | b_i.
    """
    N = 12
    MAX_M = 1000000
    DEADLINE = 85.0  # Safety margin for 90s limit
    start_time = time.time()
    
    # LCM of 1..12
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = 1
    for i in range(1, N + 1):
        M = lcm(M, i)
        
    # M is 27720
    # f(12, m) depends only on m % M
    
    def has_matching(r, L):
        """
        Check if (r, r+L] contains distinct b_1..b_12 with i | b_i.
        """
        # adj[i] = list of right-node indices j (1..L) such that (r+j) is divisible by i
        adj = [[] for _ in range(N + 1)]
        
        for j in range(1, L + 1):
            x = r + j
            for i in range(1, N + 1):
                if x % i == 0:
                    adj[i].append(j)
        
        # Kuhn's algorithm
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
        return 144 # Should not happen

    best_r = 0
    best_f = 0
    
    # Exhaustive search over one period
    # Check time periodically
    for r in range(M):
        if time.time() - start_time > DEADLINE:
            break
        f_val = compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
            
    return int(best_r)

# EVOLVE-BLOCK-END
