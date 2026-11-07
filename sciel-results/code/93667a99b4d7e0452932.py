"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct():
    N = 12
    MAX_M = 1000000
    DEADLINE = 85.0  # safety margin before 90s limit
    
    start_time = time.time()
    
    # Precompute: for each i in 1..12, we need to check divisibility
    # The key insight: f(12, m) is the least L such that (m, m+L] contains
    # distinct b_1,...,b_12 with i | b_i.
    
    # We want to maximize f(12, m) over m in [0, 1000000].
    
    # Strategy: 
    # 1. For a given m, f(12, m) can be found by binary search on L.
    # 2. To check if L works: we need a perfect matching in a bipartite graph
    #    where left side is {1,...,12} and right side is numbers in (m, m+L].
    #    Edge from i to v exists iff i divides v.
    
    # Since L is at most 144 (per the problem statement), and N=12,
    # we can use a fast matching check.
    
    # Better approach: enumerate all m values, but that's 1000001 values.
    # For each m, we need to find f(12, m).
    
    # Key observation: The pattern of divisibility is periodic with period
    # lcm(1,2,...,12) = 27720. So f(12, m) depends only on m mod 27720.
    # We can precompute f(12, m) for m in [0, 27719] and then pick the best.
    
    PERIOD = 27720
    
    def check_matching(m, L):
        """Check if there's a perfect matching for window (m, m+L]."""
        # Build adjacency: for each i in 1..12, which values in (m, m+L] are divisible by i
        # Use DFS-based bipartite matching (Hungarian-lite)
        
        # candidates[i] = list of indices into the value list
        # values = [m+1, m+2, ..., m+L]
        # But L <= 144, so at most 144 values.
        
        # For each i, find values v in (m, m+L] such that i | v
        adj = [[] for _ in range(N)]  # adj[i-1] = list of value indices
        for i in range(1, N+1):
            # First value >= m+1 that is divisible by i
            first = ((m + 1) + i - 1) // i * i
            if first < m + 1:
                first += i
            for v in range(first, m + L + 1, i):
                if v <= m + L:
                    adj[i-1].append(v - (m+1))  # 0-indexed within the window
        
        # Now do bipartite matching: left = 12 nodes, right = L nodes
        # Use DFS augmenting path
        
        match_to_right = [-1] * L  # match_to_right[j] = left node matched to right j
        match_to_left = [-1] * N   # match_to_left[i] = right node matched to left i
        
        def dfs(u, visited):
            for j in adj[u]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to_right[j] == -1 or dfs(match_to_right[j], visited):
                    match_to_right[j] = u
                    match_to_left[u] = j
                    return True
            return False
        
        count = 0
        for u in range(N):
            visited = [False] * L
            if dfs(u, visited):
                count += 1
        
        return count == N
    
    def compute_f(m):
        """Compute f(12, m) = least positive L such that matching exists."""
        # L is at most 144 per problem statement
        # Binary search on L
        lo, hi = 1, 144
        best = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if check_matching(m, mid):
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best
    
    # Precompute f for all residues mod PERIOD
    # But 27720 * (binary search * matching) might be too slow.
    # Let's estimate: for each m, binary search takes ~8 steps.
    # Each check_matching: for each i, at most L/i values. Total edges ~ L * sum(1/i) ~ 144 * 3.5 ~ 500.
    # DFS matching on 12x144 graph is fast.
    # 27720 * 8 * (small constant) should be fine.
    
    best_m = 0
    best_L = 0
    
    # We can optimize: only check m in [0, PERIOD-1], then the answer is the same for m + k*PERIOD
    # But we need m in [0, 1000000]. The best residue r gives f(12, r) = f(12, r + k*PERIOD).
    # So we just need to find the best residue.
    
    for m in range(PERIOD):
        if time.time() - start_time > DEADLINE - 5:
            break
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    # If we found a good m, return it. Otherwise return 0.
    return int(best_m)

# EVOLVE-BLOCK-END
