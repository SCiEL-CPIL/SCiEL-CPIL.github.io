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
    # Build adjacency: for each i (0..11), which positions j (0..L-1) have (m+1+j) % (i+1) == 0
    # Use bitmask per position for speed
    base = m + 1
    base_mod = base % P
    # adj[i] = list of j where position j is divisible by (i+1)
    adj = [[] for _ in range(12)]
    for j in range(L):
        res = (base_mod + j) % P
        mask = MASKS[res]
        # For each bit set in mask, add j to that i's adjacency
        mm = mask
        while mm:
            low = mm & (-mm)
            i = low.bit_length() - 1
            adj[i].append(j)
            mm ^= low
    
    # Hall's theorem check + bipartite matching via DFS (augmenting paths)
    # Try to find matching of size 12
    match_to_j = [-1] * 12  # match_to_j[i] = j assigned to i, or -1
    # Actually we need: for each i, pick a distinct j from adj[i]
    # Use standard bipartite matching: left = i (12 nodes), right = j (L nodes)
    
    # For speed, use greedy + augmenting path
    match_j = [-1] * L  # match_j[j] = i assigned to j, or -1
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_j[j] == -1 or dfs(match_j[j], visited):
                match_j[j] = i
                return True
        return False
    
    count = 0
    for i in range(12):
        visited = [False] * L
        if dfs(i, visited):
            count += 1
        else:
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
    deadline = time.time() + 85  # safety margin before 90s kill
    best_m = 12
    best_f = _f(12)
    
    # Search strategy: sample m values, focus on promising regions
    # Key: m mod P determines the pattern. Worst m likely has certain residue properties.
    # Try a range of m values with good coverage.
    
    # First, check some heuristic candidates
    candidates = [12, 0, 1, P - 1, P - 12, 27720, 27720 - 1]
    for m in candidates:
        if m < 0 or m > 1000000:
            continue
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Now do a broader search over residues mod P
    # Since pattern repeats every P, check one representative per residue class
    # But P = 27720, and we need to be efficient
    # Sample residues strategically
    
    # Try multiples and near-multiples of small numbers
    step = max(1, (1000000 // 200))  # ~5000 samples max
    for m in range(0, 1000001, step):
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Refine around the best found
    if best_f > 12:
        lo_bound = max(0, best_m - 200)
        hi_bound = min(1000000, best_m + 200)
        for m in range(lo_bound, hi_bound + 1):
            if time.time() > deadline:
                break
            f_val = _f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
