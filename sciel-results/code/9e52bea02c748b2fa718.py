import time

def _has_matching(m, L, d0_table, N):
    """Check if interval (m, m+L] contains distinct b_1..b_N with i | b_i."""
    cand_masks = [0] * (N + 1)
    cand_counts = [0] * (N + 1)
    
    for i in range(1, N + 1):
        r = m % i
        d_start = d0_table[i][r]
        if d_start > L:
            return False
        
        mask = 0
        d = d_start
        while d <= L:
            mask |= (1 << (d - 1))
            d += i
        cand_masks[i] = mask
        cand_counts[i] = mask.bit_count()
        if cand_counts[i] == 0:
            return False
    
    pos_order = list(range(1, N + 1))
    pos_order.sort(key=lambda i: cand_counts[i])
    
    # Use iterative Hopcroft-Karp style or optimized Kuhn with memo
    match_to = [-1] * (L + 1)
    
    # Precompute candidate lists for faster iteration
    cand_lists = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        mask = cand_masks[i]
        lst = []
        while mask:
            lowbit = mask & (-mask)
            d_idx = lowbit.bit_length()
            lst.append(d_idx)
            mask ^= lowbit
        cand_lists[i] = lst
    
    def dfs(pos, visited_mask):
        for d_idx in cand_lists[pos]:
            bit = 1 << (d_idx - 1)
            if visited_mask & bit:
                continue
            visited_new = visited_mask | bit
            if match_to[d_idx] == -1 or dfs(match_to[d_idx], visited_new):
                match_to[d_idx] = pos
                return True
        return False

    for pos in pos_order:
        if not dfs(pos, 0):
            return False
    return True


def construct():
    start_time = time.time()
    deadline = 88.0
    
    N = 12
    PERIOD = 27720  # lcm(1,2,...,12)
    
    # Precompute d0 for each i and residue r
    d0_table = [[0] * i for i in range(N + 1)]
    for i in range(1, N + 1):
        for r in range(i):
            d = (-r) % i
            if d == 0:
                d = i
            d0_table[i][r] = d

    best_m = 0
    best_L = 0
    
    # Search all residues modulo PERIOD
    for r in range(PERIOD):
        if time.time() - start_time > deadline:
            break
        
        # Binary search for minimal L
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if _has_matching(r, mid, d0_table, N):
                hi = mid
            else:
                lo = mid + 1
        
        if lo > best_L:
            best_L = lo
            best_m = r
            if best_L == 144:
                break

    return int(best_m)
