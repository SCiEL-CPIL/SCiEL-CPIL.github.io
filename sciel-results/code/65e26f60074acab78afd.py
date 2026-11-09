import time

P = 27720  # LCM(1..12)
MAX_L = 144

def precompute_position_masks():
    """Precompute for each i in 1..12 and each target residue t in 0..i-1,
    the bitmask of positions j in [0, MAX_L) where j % i == t."""
    pos_masks = [[0] * i for i in range(1, 13)]
    for i in range(1, 13):
        for t in range(i):
            mask = 0
            for j in range(t, MAX_L, i):
                mask |= (1 << j)
            pos_masks[i-1][t] = mask
    return pos_masks

POS_MASKS = precompute_position_masks()

def has_perfect_matching(m, L):
    """Check if (m, m+L] has a perfect matching for 1..12."""
    if L < 12:
        return False
    
    base = m + 1
    avail = [0] * 12
    
    # Precompute valid positions for each i using precomputed masks
    for i in range(1, 13):
        target_res = (-base) % i
        # Get precomputed mask and mask to L bits
        mask = POS_MASKS[i-1][target_res] & ((1 << L) - 1)
        avail[i-1] = mask
        if mask == 0:
            return False

    # Bipartite matching using iterative DFS with bitmasks
    match_to = [-1] * L  # match_to[j] = i, or -1
    
    def dfs(i, visited):
        candidates = avail[i] & ~visited
        while candidates:
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
    for L in range(12, MAX_L + 1):
        if has_perfect_matching(m, L):
            return L
    return MAX_L

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_m = 0
    best_L = 0
    
    # Search all residues 0..P-1
    for m in range(0, P):
        if time.time() > deadline:
            break
            
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
            if best_L >= MAX_L:
                return best_m
    
    return best_m
