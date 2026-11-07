import time

def _precompute_adj(m, n=12, max_L=144):
    """Precompute adjacency lists for all L up to max_L for a given m."""
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = (-m) % i
        if r == 0:
            start_k = i
        else:
            start_k = r
            
        for k in range(start_k, max_L + 1, i):
            adj[i - 1].append(k)
    return adj

def _can_match_fast(adj, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using precomputed adj."""
    # Quick check: if any i has no candidates <= L, return False
    # adj[i] is sorted
    for i in range(n):
        if not adj[i] or adj[i][0] > L:
            return False
        # Optional: check if count is sufficient? Not strictly necessary for existence, but good for pruning.
        # Actually, just checking if the list is empty up to L is enough for the "no candidates" case.
        # The DFS will handle the rest.

    match_val = [-1] * (L + 1)
    
    # Use a timestamp array for visited to avoid re-initialization?
    # L is small (<=144), so re-initializing [False]*(L+1) is fast.
    
    def dfs(i, visited):
        for k in adj[i]:
            if k > L:
                break
            if visited[k]:
                continue
            visited[k] = True
            if match_val[k] == -1 or dfs(match_val[k], visited):
                match_val[k] = i
                return True
        return False

    # Sort vertices by degree (fewest candidates <= L first)
    # This helps Kuhn's algorithm
    # We can approximate degree by the number of elements in adj[i] that are <= L
    # Since adj[i] is sorted, we can find the count via binary search or just iterate (small size)
    # n=12, so sorting 12 items is trivial.
    def count_leq(lst, limit):
        count = 0
        for k in lst:
            if k > limit:
                break
            count += 1
        return count

    order = sorted(range(n), key=lambda x: count_leq(adj[x], L))
    
    for i in order:
        visited = [False] * (L + 1)
        if not dfs(i, visited):
            return False
    return True

def _f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    adj = _precompute_adj(m, n, 144)
    
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(adj, mid, n):
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
