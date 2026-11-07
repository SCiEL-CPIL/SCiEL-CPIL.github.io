import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    N = 12
    LCM = 27720  # LCM(1,2,...,12)

    def can_match(m, L):
        """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
        # For each i, the candidate b_i values in (m, m+L] divisible by i
        # b_i must satisfy: m < b_i <= m+L and i | b_i
        # So b_i = i * k where m < i*k <= m+L
        # k in (m/i, (m+L)/i], i.e., floor(m/i)+1 <= k <= floor((m+L)/i)
        
        # Build bipartite graph: left = {1..12}, right = candidate integers
        # Edge i -> b if i | b and m < b <= m+L
        # We need a perfect matching (12 distinct b's)
        
        # Collect all candidate b values
        candidates = set()
        for i in range(1, 13):
            k_min = m // i + 1
            k_max = (m + L) // i
            for k in range(k_min, k_max + 1):
                candidates.add(i * k)
        
        if len(candidates) < 12:
            return False
        
        # Bipartite matching using DFS (Hungarian-style)
        # left nodes: 1..12, right nodes: candidate values
        # adj[i] = list of candidate b's divisible by i
        adj = {i: [] for i in range(1, 13)}
        for i in range(1, 13):
            k_min = m // i + 1
            k_max = (m+L) // i
            for k in range(k_min, k_max + 1):
                adj[i].append(i * k)
        
        # If any i has no candidates, impossible
        for i in range(1, 13):
            if not adj[i]:
                return False
        
        # DFS-based bipartite matching
        match_r = {}  # right_node -> left_node
        
        def dfs(u, visited):
            for v in adj[u]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_r or dfs(match_r[v], visited):
                    match_r[v] = u
                    return True
            return False
        
        count = 0
        for u in range(1, 13):
            visited = set()
            if dfs(u, visited):
                count += 1
            else:
                return False
        return count == 12

    def f_12(m):
        """Compute the least positive L such that (m, m+L] works."""
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # Exhaustive search over one period
    best_m = 0
    best_val = 0

    for r in range(LCM):
        if time.time() > deadline:
            break
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_m = r

    # Also check if we can use a larger m within [0, 1000000]
    # f_12(m) depends only on m mod 27720, so best_m in [0, 27719] is optimal
    # but we need m <= 1000000, which is satisfied
    
    return int(best_m)
