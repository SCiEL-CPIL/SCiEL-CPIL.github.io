import time

# EVOLVE-BLOCK-START
P = 27720  # LCM(1..12)

def _precompute_masks():
    """For each residue r in [0, P), compute bitmask of which i (1..12) divide r."""
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    return masks

MASKS = _precompute_masks()

def _has_matching(m, L):
    """Check if interval (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    if L < 12:
        return False
    
    base = m + 1
    base_mod = base % P
    
    # adj[i] = list of j where position j (value m+1+j) is divisible by (i+1)
    adj = [[] for _ in range(12)]
    for j in range(L):
        res = (base_mod + j) % P
        mask = MASKS[res]
        mm = mask
        while mm:
            low = mm & (-mm)
            i = low.bit_length() - 1
            adj[i].append(j)
            mm ^= low
            
    # Quick check: if any i has no candidates, fail
    for i in range(12):
        if not adj[i]:
            return False

    match_j = [-1] * L  # match_j[j] = i assigned to j, or -1
    seen = [0] * L      # seen[j] = timestamp of last visit
    timestamp = 0
    
    def dfs(i, ts):
        for j in adj[i]:
            if seen[j] == ts:
                continue
            seen[j] = ts
            if match_j[j] == -1 or dfs(match_j[j], ts):
                match_j[j] = i
                return True
        return False
    
    for i in range(12):
        timestamp += 1
        if not dfs(i, timestamp):
            return False
    return True

def _f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has valid matching."""
    # L must be at least 12
    lo, hi = 12, 144  # f is at most 144 per problem statement
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 85  # safety margin
    
    best_m = 0
    best_f = 0
    
    # Since f(m) depends only on m % P, we only need to check m in [0, P)
    # P = 27720
    # We iterate through all residues to find the global maximum.
    
    for m in range(P):
        if time.time() > deadline:
            break
        
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
# EVOLVE-BLOCK-END
