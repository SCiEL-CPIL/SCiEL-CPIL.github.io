import time

P = 27720  # LCM(1..12)
MAX_L = 144

def _precompute_div_mask():
    """
    div_mask[r][j] = bitmask where bit (i-1) is set if i divides (r + 1 + j)
    for r in 0..P-1, j in 0..MAX_L-1
    """
    div_mask = [[0] * MAX_L for _ in range(P)]
    for r in range(P):
        base = r + 1
        for j in range(MAX_L):
            val = base + j
            mask = 0
            for i in range(1, 13):
                if val % i == 0:
                    mask |= (1 << (i - 1))
            div_mask[r][j] = mask
    return div_mask

_DIV_MASK = _precompute_div_mask()

def _has_matching(r, L):
    """
    Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i,
    where m mod P = r.
    Uses bitmask adjacency and DFS augmenting paths.
    """
    if L < 12:
        return False
    
    # Build adj[i] as bitmask of positions j in 0..L-1 where i divides (r+1+j)
    adj = [0] * 13  # adj[i] for i=1..12
    for j in range(L):
        mask = _DIV_MASK[r][j]
        for i in range(1, 13):
            if mask & (1 << (i - 1)):
                adj[i] |= (1 << j)
    
    # Quick check: each i must have at least one candidate
    for i in range(1, 13):
        if adj[i] == 0:
            return False
    
    # DFS-based bipartite matching
    match_pos = [-1] * L  # match_pos[j] = i matched to position j, or -1
    
    def dfs(i, visited):
        candidates = adj[i]
        while candidates:
            low_bit = candidates & (-candidates)
            j = low_bit.bit_length() - 1
            candidates ^= low_bit
            if visited & low_bit:
                continue
            visited |= low_bit
            if match_pos[j] == -1 or dfs(match_pos[j], visited):
                match_pos[j] = i
                return True
        return False
    
    for i in range(1, 13):
        if not dfs(i, 0):
            return False
    return True

def _compute_f(r):
    """Compute f(12, m) where m mod P = r."""
    # Lower bound: L must be at least the max first-valid-position for each i
    min_L = 0
    for i in range(1, 13):
        # Find smallest j >= 0 such that i divides (r + 1 + j)
        # (r + 1 + j) % i == 0  =>  j % i == (-r-1) % i
        rem = (-r - 1) % i
        if rem == 0:
            first_j = 0
        else:
            first_j = i - rem
        # j is 0-indexed, so position in interval is j+1
        # L must be at least first_j + 1
        if first_j + 1 > min_L:
            min_L = first_j + 1
    
    lo = max(min_L, 12)
    hi = MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(r, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Search all residues mod P
    for r in range(P):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
            if best_f >= MAX_L:
                break
    
    return int(best_m)
