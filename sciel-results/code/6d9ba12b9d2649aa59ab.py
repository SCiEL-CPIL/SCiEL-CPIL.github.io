import time

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i."""
    # Build adjacency: for each i (0-indexed), list of candidate values k (1..L)
    # such that (m+k) is divisible by i+1
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        # We need k in [1, L] such that (m + k) % i == 0
        # k % i == (-m) % i
        r = (-m) % i
        if r == 0:
            start_k = i
        else:
            start_k = r
            
        # Add candidates
        for k in range(start_k, L + 1, i):
            adj[i - 1].append(k)
    
    # If any i has no candidates, return False
    for i in range(n):
        if not adj[i]:
            return False
            
    # Kuhn's algorithm for bipartite matching
    # match_val[k] = i (0-indexed) or -1
    match_val = [-1] * (L + 1)
    
    def dfs(i, visited):
        for k in adj[i]:
            if visited[k]:
                continue
            visited[k] = True
            if match_val[k] == -1 or dfs(match_val[k], visited):
                match_val[k] = i
                return True
        return False
        
    # Sort vertices by degree (fewest candidates first) to help Kuhn's
    # We iterate in this order for the main matching loop
    order = sorted(range(n), key=lambda x: len(adj[x]))
    
    for i in order:
        visited = [False] * (L + 1)
        if not dfs(i, visited):
            return False
    return True

def _f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L
    lo, hi = 1, 144  # f is at most 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 12
    period = 27720  # lcm(1, 2, ..., 12)
    
    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return best_m
