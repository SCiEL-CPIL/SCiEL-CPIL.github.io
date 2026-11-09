import time

# EVOLVE-BLOCK-START
P = 27720  # lcm(1..12)

def _has_matching(m, L):
    """Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i.
    Positions are m+1, m+2, ..., m+L (L positions).
    We need to assign each i in 1..12 to a distinct position j in 1..L such that i | (m+j).
    """
    if L < 12:
        return False
    
    # Compute valid positions for each i efficiently using modular arithmetic
    valid = []
    for i in range(1, 13):
        r = (-m) % i
        if r == 0:
            positions = list(range(i, L + 1, i))
        else:
            positions = list(range(r, L + 1, i))
        if not positions:
            return False
        valid.append((i, positions))
    
    # Sort by fewest options first for better pruning
    valid.sort(key=lambda x: len(x[1]))
    
    # DFS backtracking with used-position tracking
    used = [False] * (L + 1)
    
    def dfs(idx):
        if idx == 12:
            return True
        _, positions = valid[idx]
        for j in positions:
            if not used[j]:
                used[j] = True
                if dfs(idx + 1):
                    return True
                used[j] = False
        return False
    
    return dfs(0)

def _compute_f(m):
    """Compute f(12, m): least positive L such that matching exists."""
    if not _has_matching(m, 144):
        return 144
    
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    best_m = 0
    best_f = 0
    
    # Search all m in [0, P) since f is periodic with period P
    for m in range(P):
        if time.time() > deadline:
            break
        
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
