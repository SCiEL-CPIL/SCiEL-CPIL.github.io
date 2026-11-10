"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12

def _min_multiple_gt(m, i):
    """Smallest multiple of i strictly greater than m."""
    r = m % i
    if r == 0:
        return m + i
    else:
        return m + (i - r)

def _compute_f(m):
    """Compute f(N, m): least positive L such that (m, m+L] contains distinct b_1..b_N with i | b_i."""
    # For each i, the set of valid b_i in (m, m+L] are multiples of i in that range.
    # We need to assign distinct values.
    # Greedy: for each i, the smallest available multiple of i > m.
    # But greedy by index order may not be optimal; we need a bipartite matching or careful greedy.
    # Since L is small (<=144), we can try increasing L and check if a matching exists.
    
    # For a given L, valid b_i for index i are multiples of i in (m, m+L].
    # We need a system of distinct representatives.
    
    # Use Hall's theorem / bipartite matching.
    # The universe of numbers is (m, m+L], i.e., m+1 .. m+L.
    # For each i, the candidates are {k : m < k <= m+L, i | k}.
    
    # We'll do binary search on L, but L is at most ~144, so linear search is fine.
    # Actually, let's just try L from 1 upward.
    
    for L in range(1, 200):
        # Build candidate sets
        # candidates[i] = list of values in (m, m+L] divisible by i
        # Use a matching: assign each i a distinct value.
        # Since N=12 and L<=200, we can use a simple backtracking or bipartite matching.
        
        # Bipartite matching: left side = indices 1..N, right side = values m+1..m+L
        # Edge if value % i == 0.
        
        # Use DFS-based bipartite matching (Kuhn's algorithm).
        # Right side size is L, left side is N=12.
        
        # match_to[i] = value assigned to index i, or -1
        # match_val[v] = index assigned to value v, or -1
        
        # Actually, let's represent values as offsets 1..L (value = m + offset)
        
        adj = [[] for _ in range(N)]
        for i in range(1, N+1):
            # multiples of i in (m, m+L]
            # smallest multiple of i > m
            start = _min_multiple_gt(m, i)
            for val in range(start, m + L + 1, i):
                offset = val - m  # 1..L
                adj[i-1].append(offset)
        
        # Kuhn's algorithm
        match_val = [-1] * (L + 1)  # match_val[offset] = index (0-based) or -1
        
        def dfs(idx, visited):
            for off in adj[idx]:
                if visited[off]:
                    continue
                visited[off] = True
                if match_val[off] == -1 or dfs(match_val[off], visited):
                    match_val[off] = idx
                    return True
            return False
        
        for idx in range(N):
            visited = [False] * (L + 1)
            if not dfs(idx, visited):
                break
        else:
            # All matched
            return L
    
    return -1  # Should not reach here

def construct():
    deadline = time.time() + 80  # 80 seconds, leave 10s margin
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Search strategy: 
    # The worst m tends to be where many i have m%i == 0 (so b_i = m+i, large gaps)
    # or where collisions force extra steps.
    # Let's search all m in [0, 1000000] but that's 1M values, each taking ~L*N^2 time.
    # With L~200 and N=12, each check is ~200*12*12 = 28800 ops, times 1M = 28.8B, too slow.
    
    # Better: sample intelligently. The function f(12,m) is periodic-ish.
    # Actually, f depends on m mod lcm(1..12) = 27720. So f(12, m) = f(12, m mod 27720) for the gap structure?
    # Not exactly, because the absolute values matter for distinctness, but the pattern of which multiples
    # fall in (m, m+L] depends on m mod i for each i, which is determined by m mod lcm(1..12).
    # So f(12, m) is periodic with period lcm(1..12) = 27720.
    
    # So we only need to check m in [0, 27719]!
    
    lcm_val = 27720
    for m in range(0, lcm_val):
        if time.time() > deadline - 5:
            break
        f = _compute_f(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
