import time

def construct():
    start_time = time.time()
    deadline = 88  # 88s safety margin for 90s limit
    
    N = 12
    PERIOD = 27720  # lcm(1,2,...,12)
    
    # Precompute d0 for each i and residue r
    # d0[i][r] is the smallest d in [1, i] such that (r + d) % i == 0
    d0_table = [[0] * i for i in range(N + 1)]
    for i in range(1, N + 1):
        for r in range(i):
            # We need d such that r + d ≡ 0 (mod i)
            # d ≡ -r (mod i)
            d = (-r) % i
            if d == 0:
                d = i
            d0_table[i][r] = d

    def has_matching(m, L):
        # Build candidate masks for each position i
        # Mask bit j (0-indexed) corresponds to offset d = j + 1
        cand_masks = [0] * (N + 1)
        
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
        
        # Bipartite Matching
        # match_to[d_idx] = position i
        match_to = [-1] * L
        
        # Order positions by scarcity of candidates
        pos_order = list(range(1, N + 1))
        # Count bits in mask
        pos_order.sort(key=lambda i: bin(cand_masks[i]).count('1'))
        
        # If any position has 0 candidates, fail
        if bin(cand_masks[pos_order[0]]).count('1') == 0:
            return False

        def dfs(pos, visited_mask):
            mask = cand_masks[pos]
            m_mask = mask
            while m_mask:
                lowbit = m_mask & (-m_mask)
                d_idx = lowbit.bit_length() - 1
                m_mask ^= lowbit
                
                if visited_mask & (1 << d_idx):
                    continue
                
                visited_new = visited_mask | (1 << d_idx)
                if match_to[d_idx] == -1 or dfs(match_to[d_idx], visited_new):
                    match_to[d_idx] = pos
                    return True
            return False

        for pos in pos_order:
            if not dfs(pos, 0):
                return False
        return True

    best_m = 0
    best_L = 0
    
    for r in range(PERIOD):
        if time.time() - start_time > deadline:
            break
        
        # Binary search for minimum L
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo > best_L:
            best_L = lo
            best_m = r
            if best_L == 144:
                break

    return int(best_m)
