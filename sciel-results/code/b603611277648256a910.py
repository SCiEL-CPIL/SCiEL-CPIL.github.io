"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
P = 27720  # lcm(1..12)
MAX_L = 144

def _precompute_masks():
    """Precompute divisibility masks for each residue r in [0, P).
    masks[r] is a list of MAX_L integers, where masks[r][j] is a 12-bit mask
    with bit i set if (i+1) divides (r+1+j)."""
    masks = []
    for r in range(P):
        row = []
        base = r + 1
        for j in range(MAX_L):
            val = base + j
            m = 0
            # i=1: always divides
            m |= (1 << 0)
            if val % 2 == 0: m |= (1 << 1)
            if val % 3 == 0: m |= (1 << 2)
            if val % 4 == 0: m |= (1 << 3)
            if val % 5 == 0: m |= (1 << 4)
            if val % 6 == 0: m |= (1 << 5)
            if val % 7 == 0: m |= (1 << 6)
            if val % 8 == 0: m |= (1 << 7)
            if val % 9 == 0: m |= (1 << 8)
            if val % 10 == 0: m |= (1 << 9)
            if val % 11 == 0: m |= (1 << 10)
            if val % 12 == 0: m |= (1 << 11)
            row.append(m)
        masks.append(row)
    return masks

MASKS = _precompute_masks()

def _has_matching(row, L):
    """Check if window of length L starting at position 0 (relative) has a perfect matching.
    row[j] has bit i set if (i+1) divides position j+1."""
    if L < 12:
        return False
    
    # Build adjacency: adj[i] = list of j in [0, L) where bit i is set in row[j]
    adj = [[] for _ in range(N)]
    for j in range(L):
        mask = row[j]
        for i in range(N):
            if mask & (1 << i):
                adj[i].append(j)
    
    # Kuhn's algorithm (DFS augmenting path)
    match_to_pos = [-1] * L  # match_to_pos[j] = i or -1
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to_pos[j] == -1 or dfs(match_to_pos[j], visited):
                match_to_pos[j] = i
                return True
        return False
    
    for i in range(N):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _compute_f_for_residue(r):
    """Compute f(12, r) for residue r in [0, P)."""
    row = MASKS[r]
    # Binary search for minimum L
    lo, hi = 12, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(row, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """Return the optimal m in [0, 1000000] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_m = 0
    best_f = 0
    
    for r in range(P):
        if time.time() > deadline:
            break
        
        f_val = _compute_f_for_residue(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
            # If we hit the theoretical max, we can stop
            if best_f == MAX_L:
                break
    
    # best_m is in [0, P-1] which is within [0, 1000000]
    return int(best_m)

# EVOLVE-BLOCK-END
