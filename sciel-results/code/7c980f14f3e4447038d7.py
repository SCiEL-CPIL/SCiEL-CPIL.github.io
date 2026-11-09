import time

# EVOLVE-BLOCK-START
N = 12
P = 27720  # LCM(1..12)
DEADLINE = time.time() + 75  # safety margin before 90s kill

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
    # adj[i] = list of positions j (0..L-1) where (m+1+j) is divisible by (i+1)
    adj = [[] for _ in range(N)]
    for j in range(L):
        pos = (base_mod + j) % P
        mask = MASKS[pos]
        # For each i that divides this position, add j to adj[i]
        for i in range(N):
            if mask & (1 << i):
                adj[i].append(j)
    
    # Bipartite matching: match each i (0..11) to a distinct position j
    # Use DFS-based augmenting path
    match_to_pos = [-1] * N  # match_to_pos[i] = position assigned to i, or -1
    pos_matched_by = [-1] * L  # pos_matched_by[j] = i assigned to position j, or -1
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if pos_matched_by[j] == -1 or dfs(pos_matched_by[j], visited):
                pos_matched_by[j] = i
                match_to_pos[i] = j
                return True
        return False
    
    for i in range(N):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has valid matching."""
    # Binary search on L
    lo, hi = 12, 144  # L is at most 144 per problem statement
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    best_m = 12
    best_f = _compute_f(12)
    
    # Search over all residues in one period
    for m in range(P):
        if time.time() > DEADLINE:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
