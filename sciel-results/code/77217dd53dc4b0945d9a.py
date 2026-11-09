import time

def _has_perfect_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
    # Build adjacency: for each i in 1..n, which positions j (1..L) have (m+j) % i == 0
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        target = (-m) % i
        if target == 0:
            first = i
        else:
            first = target
        if first > L:
            return False
        j = first
        while j <= L:
            mask |= (1 << (j - 1))
            j += i
        adj[i] = mask
    
    # DFS matching with bit operations, most constrained first
    match_r = [0] * L
    
    # Precompute candidate counts for ordering
    counts = [bin(adj[i]).count('1') for i in range(1, n + 1)]
    order = list(range(1, n + 1))
    order.sort(key=lambda i: counts[i - 1])
    
    def dfs(i, visited):
        candidates = adj[i] & ~visited
        while candidates:
            low_bit = candidates & -candidates
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            if match_r[j_idx] == 0 or dfs(match_r[j_idx], visited | low_bit):
                match_r[j_idx] = i
                return True
        return False
    
    for i in order:
        if not dfs(i, 0):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    max_m = 1000000
    P = 27720  # LCM(1..12)
    
    best_m = 0
    best_f = 0
    
    # Precompute f for all residues r in 0..P-1
    # f(12, m) depends only on m mod P
    for r in range(P):
        if time.time() > deadline:
            break
        
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    best_m = int(best_m)
    if best_m < 0 or best_m > max_m:
        best_m = 0
    
    return best_m
