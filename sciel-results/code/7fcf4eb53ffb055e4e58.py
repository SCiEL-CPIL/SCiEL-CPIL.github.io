import time

P = 27720  # LCM(1..12)

def precompute_divisibility():
    """For each residue r in 0..P-1, store a bitmask of which i in 1..12 divide r."""
    div_mask = [0] * P
    divisors = [[] for _ in range(P)]
    for r in range(P):
        mask = 0
        for i in range(1, 13):
            if r % i == 0:
                mask |= (1 << (i - 1))
                divisors[r].append(i - 1)
        div_mask[r] = mask
    return div_mask, divisors

DIV_MASK, DIVISORS = precompute_divisibility()

def has_perfect_matching(m, L):
    """Check if (m, m+L] has a perfect matching for 1..12."""
    base = m + 1
    # Build adjacency: for each i (0..11), list of positions j where i+1 divides base+j
    adj = [[] for _ in range(12)]
    
    start_res = base % P
    for j in range(L):
        r = (start_res + j) % P
        for i in DIVISORS[r]:
            adj[i].append(j)
    
    # Bipartite matching using DFS (Kuhn's algorithm)
    # Process left nodes in order of increasing degree (most constrained first)
    order = sorted(range(12), key=lambda i: len(adj[i]))
    
    match_to = [-1] * L
    match_from = [-1] * 12
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to[j] == -1 or dfs(match_to[j], visited):
                match_to[j] = i
                match_from[i] = j
                return True
        return False
    
    for i in order:
        visited = [False] * L
        if not dfs(i, visited):
            return False
    
    return True

def compute_f(m):
    """Compute f(12, m) = least L such that (m, m+L] has perfect matching."""
    # Binary search for minimum L
    lo, hi = 12, 144  # L must be at least 12 (need 12 distinct positions), at most 144
    if not has_perfect_matching(m, hi):
        hi = 200
        if not has_perfect_matching(m, hi):
            return hi  # Fallback
    while lo < hi:
        mid = (lo + hi) // 2
        if has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_m = 0
    best_L = 0
    
    # Only need to check m in 0..P-1 since f(12, m) is periodic with period P
    # But m can be 0..1000000, and f depends only on m mod P
    # So we check all residues 0..P-1
    
    # To speed up, process in chunks and check time
    chunk_size = 500
    for m_start in range(0, P, chunk_size):
        m_end = min(m_start + chunk_size, P)
        for m in range(m_start, m_end):
            if time.time() > deadline:
                break
            
            L = compute_f(m)
            if L > best_L:
                best_L = L
                best_m = m
                if best_L >= 144:
                    # Can't do better
                    return best_m
        if time.time() > deadline:
            break
    
    return best_m
