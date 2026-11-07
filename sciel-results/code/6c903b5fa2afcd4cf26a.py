import time
import sys

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i."""
    # For each i in 1..n, find multiples of i in (m, m+L]
    # Use bipartite matching (Kuhn's algorithm)
    
    # Build adjacency: for each i (0-indexed), list of candidate values
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        # First multiple of i > m
        first = ((m // i) + 1) * i
        end = m + L
        v = first
        while v <= end:
            adj[i - 1].append(v)
            v += i
    
    # If any i has no candidates, return False
    for i in range(n):
        if not adj[i]:
            return False
    
    # Match right nodes: match_val maps value -> i (0-indexed)
    match_val = {}
    
    def dfs(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            # If v is unmatched or the previous match can be reassigned
            if v not in match_val or dfs(match_val[v], visited):
                match_val[v] = i
                return True
        return False
    
    for i in range(n):
        visited = set()
        if not dfs(i, visited):
            return False
    
    return True

def _f(m, n=12):
    """Find the least positive L such that (m, m+L] contains distinct b_1..b_n with i | b_i."""
    # Binary search on L
    # L must be at least 1
    # Upper bound: try up to some reasonable limit
    # For n=12, L is at most around 144 per the problem statement
    lo, hi = 1, 200
    # First check if L=hi works
    if not _can_match(m, hi, n):
        # Try larger
        hi = 400
        if not _can_match(m, hi, n):
            hi = 800
            if not _can_match(m, hi, n):
                return 0
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 80  # 80 second deadline, leaving 10s margin
    
    best_m = 0
    best_f = _f(0)
    
    # Try a range of m values, prioritizing those likely to give high f
    # f(m) is the "worst interval starts" - we want to maximize it
    # Let's try various m values
    
    # First, do a coarse search over m
    # We'll sample m values and compute f(m)
    
    # Start with some heuristic candidates
    candidates = []
    
    # Try m values that are multiples of large numbers, or just after them
    # For n=12, the constraint involves divisors 1-12
    # lcm(1,2,...,12) = 27720
    
    # Let's try a grid of m values
    step = 100
    for m in range(0, 1000001, step):
        if time.time() > deadline - 5:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # If we have time, do a finer search around the best m found
    if time.time() < deadline - 10:
        # Fine search around best_m
        search_range = 500
        for m in range(max(0, best_m - search_range), min(1000001, best_m + search_range)):
            if time.time() > deadline - 3:
                break
            f_val = _f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    # Additional targeted search: try m values that are 1 less than multiples of 27720
    if time.time() < deadline - 5:
        lcm12 = 27720
        for k in range(1, 40):
            m = k * lcm12 - 1
            if m < 0 or m > 1000000:
                continue
            f_val = _f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
