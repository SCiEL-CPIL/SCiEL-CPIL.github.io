import time

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin under 90s
    
    N = 12
    PERIOD = 27720  # lcm(1,2,...,12)
    
    # Precompute: for each residue r in 0..PERIOD-1, and for each position i in 1..12,
    # the offset pattern of multiples. But more efficiently, we can precompute
    # for each i, the set of offsets j (mod PERIOD) where i divides (r + j).
    # Actually, since the interval is (m, m+L], and m can be any value,
    # the key is: for a given m and L, which values in (m, m+L] are divisible by i?
    # Value v is in (m, m+L] iff m < v <= m+L, i.e., v - m in (0, L].
    # v is divisible by i iff v mod i == 0.
    # Let v = m + d, where 1 <= d <= L.
    # m + d ≡ 0 (mod i) iff d ≡ -m (mod i) iff d ≡ (i - m % i) % i (mod i).
    # So for each i, the valid offsets d are: d ≡ (-m) mod i, with 1 <= d <= L.
    # The smallest such d: d0 = ((i - m % i) % i). If d0 == 0, d0 = i.
    # Then valid d's are: d0, d0+i, d0+2i, ... <= L.
    
    def has_matching(m, L):
        """Check if there's a perfect matching for interval (m, m+L]."""
        # For each position i (1..12), compute the set of valid offsets d in [1, L]
        # such that i divides (m + d).
        # Represent each position's candidate set as a bitmask over offsets 1..L.
        
        # Precompute candidate bitmasks for each i
        cand_masks = [0] * (N + 1)  # 1-indexed
        for i in range(1, N + 1):
            # Find smallest d in [1, L] with m + d ≡ 0 (mod i)
            rem = m % i
            d0 = (i - rem) % i
            if d0 == 0:
                d0 = i
            if d0 > L:
                return False  # No candidate for position i
            # Set bits for d0, d0+i, d0+2i, ... <= L
            mask = 0
            d = d0
            while d <= L:
                mask |= (1 << (d - 1))  # bit position d-1 for offset d
                d += i
            cand_masks[i] = mask
        
        # Now do bipartite matching using bitmask representation
        # Left side: positions 1..12, Right side: offsets 1..L
        # match_to[d-1] = position i assigned to offset d, or -1
        match_to = [-1] * L
        
        # Order positions by number of candidates (ascending) for early failure
        pos_order = list(range(1, N + 1))
        pos_order.sort(key=lambda i: bin(cand_masks[i]).count('1'))
        
        # If any position has 0 candidates, already returned False above
        
        def dfs(pos, visited_mask):
            """Try to assign position pos to some unvisited offset."""
            mask = cand_masks[pos]
            # Try each candidate offset for this position
            m = mask
            while m:
                # Get lowest set bit
                lowbit = m & (-m)
                d_idx = lowbit.bit_length() - 1  # 0-indexed offset index
                m ^= lowbit
                if visited_mask & (1 << d_idx):
                    continue
                visited_mask_new = visited_mask | (1 << d_idx)
                if match_to[d_idx] == -1 or dfs(match_to[d_idx], visited_mask_new):
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
                break  # Can't do better than 144
    
    return int(best_m)
