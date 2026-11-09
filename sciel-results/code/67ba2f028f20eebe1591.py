import time

# EVOLVE-BLOCK-START
P = 27720  # lcm(1..12)

def _compute_f_fast(m_res):
    """Compute f(12, m) for m ≡ m_res (mod P) using precomputed adjacency and fast matching."""
    # Precompute valid positions for each i in 1..12 for j=1..144
    # (m_res + j) % i == 0
    valid = [[] for _ in range(13)]  # 1-indexed, index 0 unused
    for i in range(1, 13):
        # j such that (m_res + j) % i == 0, i.e., j % i == (-m_res) % i
        target = (-m_res) % i
        for j in range(1, 145):
            if j % i == target:
                valid[i].append(j)
        if not valid[i]:
            return 144  # No position for i even at L=144
    
    def has_matching(L):
        # Quick check: each i must have at least one valid position in [1..L]
        for i in range(1, 13):
            # valid[i] is sorted, so check if first element <= L
            if not valid[i] or valid[i][0] > L:
                return False
            # More precisely, count how many are <= L
            # Since valid[i] is sorted, we can binary search or just check
            cnt = 0
            for j in valid[i]:
                if j > L:
                    break
                cnt += 1
            if cnt == 0:
                return False
        
        # Kuhn's algorithm with ordering by fewest options
        # Precompute counts for ordering
        order = []
        for i in range(1, 13):
            cnt = 0
            for j in valid[i]:
                if j > L:
                    break
                cnt += 1
            order.append((cnt, i))
        order.sort()
        
        match_pos = [-1] * (L + 1)
        
        def dfs(i, visited):
            for j in valid[i]:
                if j > L or visited[j]:
                    continue
                visited[j] = True
                if match_pos[j] == -1 or dfs(match_pos[j], visited):
                    match_pos[j] = i
                    return True
            return False
        
        for cnt, i in order:
            visited = [False] * (L + 1)
            if not dfs(i, visited):
                return False
        return True
    
    # Binary search for minimum L
    lo, hi = 12, 144
    if not has_matching(144):
        return 144
    while lo < hi:
        mid = (lo + hi) // 2
        if has_matching(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    best_m = 0
    best_f = 0
    
    # Search all m in [0, P) since f is periodic with period P
    for m in range(P):
        if time.time() > deadline:
            break
        
        f_val = _compute_f_fast(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Return best_m as Python int
    return int(best_m)
# EVOLVE-BLOCK-END
