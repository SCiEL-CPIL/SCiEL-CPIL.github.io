import time

P = 27720  # LCM(1..12)

def precompute_divisibility():
    """For each residue r in 0..P-1, store a bitmask of which i in 1..12 divide r."""
    div_mask = [0] * P
    for r in range(P):
        mask = 0
        for i in range(1, 13):
            if r % i == 0:
                mask |= (1 << (i - 1))
        div_mask[r] = mask
    return div_mask

DIV_MASK = precompute_divisibility()

def has_perfect_matching(m, L):
    """Check if (m, m+L] has a perfect matching for 1..12 using bitmask optimization."""
    if L < 12:
        return False
    
    base = m + 1
    start_res = base % P
    
    # avail[i] will be a bitmask of positions j (0..L-1) where i divides base+j
    avail = [0] * 12
    
    # Precompute valid positions for each i
    for i in range(1, 13):
        # We need j such that (base + j) % i == 0
        # j % i == (-base) % i
        target_res = (-base) % i
        mask = 0
        # Iterate through valid j in [0, L)
        # j = target_res + k*i
        first_j = target_res
        if first_j >= L:
            # No valid position
            return False
        # Build mask
        # Using a loop is fine since L <= 144
        for j in range(first_j, L, i):
            mask |= (1 << j)
        avail[i-1] = mask
        
        # Quick check: if no positions, fail
        if mask == 0:
            return False

    # Bipartite matching using DFS with bitmasks
    match_to = [-1] * L  # match_to[j] = i, or -1
    
    # SystemRecursionError guard? N=12, depth max 12, safe.
    def dfs(i, visited):
        candidates = avail[i] & ~visited
        while candidates:
            # Get lowest set bit
            lsb = candidates & -candidates
            j = lsb.bit_length() - 1
            candidates ^= lsb
            
            if visited & lsb:
                continue
            
            visited |= lsb
            
            if match_to[j] == -1 or dfs(match_to[j], visited):
                match_to[j] = i
                return True
        return False

    for i in range(12):
        visited = 0
        if not dfs(i, visited):
            return False
            
    return True

def compute_f(m):
    """Compute f(12, m) = least L such that (m, m+L] has perfect matching."""
    # We know L is at most 144.
    # We can scan from 12 upwards.
    # This is faster than binary search if the answer is small, 
    # and avoids log factor overhead.
    for L in range(12, 145):
        if has_perfect_matching(m, L):
            return L
    return 144

def construct():
    start_time = time.time()
    # Deadline guard: 90s limit. Use 85s to be safe.
    deadline = start_time + 85.0
    
    best_m = 0
    best_L = 0
    
    # Search all residues 0..P-1
    # Optimization: Check time periodically.
    # We can also try to jump to promising residues if we had a heuristic, 
    # but exact search is required to guarantee max.
    
    # To speed up, we can process in chunks and check time
    chunk_size = 100
    for m_start in range(0, P, chunk_size):
        m_end = min(m_start + chunk_size, P)
        for m in range(m_start, m_end):
            if time.time() > deadline:
                return best_m
            
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
