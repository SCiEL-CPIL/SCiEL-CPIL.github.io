import time

def construct():
    start_time = time.time()
    limit = 85  # Leave 5s for overhead

    N = 12
    period = 27720
    max_L = 144

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def has_perfect_matching(masks, N, L):
        """Check if there's a perfect matching using DFS augmenting paths (Hopcroft-Karp lite)."""
        # masks[i] is a bitmask of valid positions for i (1-indexed)
        # We try to assign each i to a distinct position.
        # Use DFS-based augmenting path matching.
        
        # pos_match[pos] = which i is matched to position pos, or -1
        # pos ranges 0..L-1
        pos_match = [-1] * L
        
        def dfs(i, seen):
            m = masks[i]
            # Iterate over set bits in m
            mm = m
            while mm:
                lsb = mm & (-mm)
                pos = lsb.bit_length() - 1  # 0-indexed position
                mm ^= lsb
                if seen[pos]:
                    continue
                seen[pos] = True
                if pos_match[pos] == -1 or dfs(pos_match[pos], seen):
                    pos_match[pos] = i
                    return True
            return False
        
        for i in range(1, N + 1):
            if masks[i] == 0:
                return False
            seen = [False] * L
            if not dfs(i, seen):
                return False
        return True

    # Precompute mask_table[L][i][r]: bitmask of valid x-1 positions for given L, i, residue r
    mask_table = [None] * (max_L + 1)
    for L in range(max_L + 1):
        mask_table[L] = [None] * (N + 1)
        for i in range(1, N + 1):
            mask_table[L][i] = [0] * i
            for r in range(i):
                mask = 0
                if r == 0:
                    start_x = i
                else:
                    start_x = r
                if start_x <= L:
                    for x in range(start_x, L + 1, i):
                        mask |= (1 << (x - 1))
                mask_table[L][i][r] = mask

    def compute_f(m):
        """Compute the least L >= 1 such that there's a perfect matching."""
        for L in range(1, max_L + 1):
            masks = [0] * (N + 1)
            feasible = True
            for i in range(1, N + 1):
                r = (-m) % i
                mask = mask_table[L][i][r]
                if mask == 0:
                    feasible = False
                    break
                masks[i] = mask
            
            if not feasible:
                continue
            
            if has_perfect_matching(masks, N, L):
                return L
        
        return max_L

    # Full scan: compute f for all 27720 residues, pick the best
    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
