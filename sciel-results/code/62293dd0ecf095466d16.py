import time
import sys

def construct():
    start_time = time.time()
    deadline = 50.0  # Leave 10s margin for evaluator overhead
    
    n = 12
    
    # Kuhn's algorithm for bipartite matching
    # left: 0..n-1 representing indices 1..n
    # right: 0..L-1 representing values m+1 .. m+L
    def has_matching(m, L):
        # right_match[j] = i if right node j is matched to left node i, else -1
        right_match = [-1] * L
        
        def dfs(u, visited):
            # u is left node index (0-based, represents i = u+1)
            i = u + 1
            # iterate over possible right nodes
            # right node j corresponds to value b = m + 1 + j
            # condition: b % i == 0
            for j in range(L):
                b = m + 1 + j
                if b % i == 0 and not visited[j]:
                    visited[j] = True
                    # if right node j is free, or the left node currently matched to j can find another match
                    if right_match[j] == -1 or dfs(right_match[j], visited):
                        right_match[j] = u
                        return True
            return False
        
        match_count = 0
        for u in range(n):
            visited = [False] * L
            if dfs(u, visited):
                match_count += 1
        return match_count == n

    def compute_f(m):
        # Binary search for smallest L
        # Lower bound: n (since we need n distinct numbers)
        # Upper bound: 144 (from problem statement max)
        lo = n
        hi = 144
        best_L = hi
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                best_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best_L

    best_m = 0
    best_f = 0
    
    # Scan a range of m
    # We expect f(12,m) to be around 12-20.
    # Scanning 0..5000 should be fast.
    # Each compute_f does ~7 binary search steps.
    # Each has_matching does ~12 * 12 * L ops. L~20. ~3000 ops.
    # 7 * 3000 = 21000 ops per m.
    # 5000 m's -> 100 million ops. Might be tight in Python in 50s?
    # Python is slow. 10^8 simple ops is ~10s.
    # Let's optimize has_matching.
    
    # Optimized has_matching:
    # Precompute multiples?
    # For a fixed m and L, the set of valid b for each i is fixed.
    
    def has_matching_fast(m, L):
        # Build adjacency list for left nodes
        adj = [[] for _ in range(n)]
        for i in range(1, n+1):
            # First multiple of i > m
            first = ((m // i) + 1) * i
            # Last multiple of i <= m+L
            last = (m + L) // i * i
            # Add indices j such that m+1+j is multiple of i
            # b = m+1+j. j = b - m - 1.
            for b in range(first, last + 1, i):
                j = b - m - 1
                if 0 <= j < L:
                    adj[i-1].append(j)
                    
        right_match = [-1] * L
        
        def dfs(u, visited):
            i = u + 1
            for j in adj[u]:
                if not visited[j]:
                    visited[j] = True
                    if right_match[j] == -1 or dfs(right_match[j], visited):
                        right_match[j] = u
                        return True
            return False
        
        match_count = 0
        for u in range(n):
            visited = [False] * L
            if dfs(u, visited):
                match_count += 1
        return match_count == n

    def compute_f_fast(m):
        lo = n
        hi = 144
        best_L = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching_fast(m, mid):
                best_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best_L

    # Scan
    # Start with small m, then maybe randomize or expand range if time permits
    m_limit = 2000
    for m in range(m_limit):
        if time.time() > deadline:
            break
        f_val = compute_f_fast(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    # If we have time, scan more or random
    while time.time() < deadline - 5:
        # Random sampling
        import random
        m_rand = random.randint(0, 1000000)
        f_val = compute_f_fast(m_rand)
        if f_val > best_f:
            best_f = f_val
            best_m = m_rand

    return int(best_m)
