"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _precompute_divisibility():
    """For each residue r in 0..27719 and offset j in 0..143, 
    compute bitmask of which i (1..12) divide (r+1+j)."""
    P = 27720
    MAX_L = 144
    # Vectorized: values[r, j] = r + 1 + j
    values = np.arange(P)[:, None] + np.arange(MAX_L)[None, :] + 1
    # div_mask[r, j] = bitmask where bit i-1 is set if i divides values[r, j]
    div_mask = np.zeros((P, MAX_L), dtype=np.int32)
    for i in range(1, 13):
        div_mask[values % i == 0] |= (1 << (i - 1))
    return div_mask

def _has_matching(div_mask_r, L):
    """Check if there's a matching of size 12 in window of length L.
    div_mask_r[j] has bit i-1 set if i divides position j (0-indexed).
    We need to assign each i (1..12) to a distinct position j in 0..L-1 
    such that i divides position j's value."""
    # Build adjacency: adj[i] is bitmask of positions where i divides
    adj = [0] * 13
    for j in range(L):
        mask = div_mask_r[j]
        for i in range(1, 13):
            if mask & (1 << (i - 1)):
                adj[i] |= (1 << j)
    
    # Standard DFS-based bipartite matching
    match_to_pos = [-1] * L  # match_to_pos[j] = i matched to position j, or -1
    
    def dfs(i, visited):
        """Try to find augmenting path from left node i"""
        bits = adj[i]
        while bits:
            j = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            if not visited[j]:
                visited[j] = True
                if match_to_pos[j] == -1 or dfs(match_to_pos[j], visited):
                    match_to_pos[j] = i
                    return True
        return False
    
    for i in range(1, 13):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _compute_f(r, div_mask):
    """Compute f(12, m) where m mod 27720 = r.
    Returns the least L such that matching exists."""
    # Binary search on L: find minimum L where matching exists
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(div_mask[r], mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    div_mask = _precompute_divisibility()
    
    best_m = 0  # fallback
    best_f = 0
    
    # First pass: quick scan to find good candidates
    # Use a subset of residues for initial scan
    candidates = []
    for r in range(0, 27720, 7):  # every 7th residue for speed
        if time.time() - start_time > DEADLINE:
            break
        f_val = _compute_f(r, div_mask)
        if f_val > best_f:
            best_f = f_val
            best_m = r
        if f_val >= 20:  # promising candidate
            candidates.append((r, f_val))
    
    # Second pass: refine around best candidates
    candidates.sort(key=lambda x: -x[1])
    for r, f_val in candidates[:50]:
        if time.time() - start_time > DEADLINE:
            break
        # Search neighborhood
        for dr in range(-5, 6):
            rr = r + dr
            if 0 <= rr < 27720:
                f_val2 = _compute_f(rr, div_mask)
                if f_val2 > best_f:
                    best_f = f_val2
                    best_m = rr
    
    # Third pass: full scan if time permits
    if time.time() - start_time < DEADLINE - 10:
        for r in range(27720):
            if time.time() - start_time > DEADLINE:
                break
            f_val = _compute_f(r, div_mask)
            if f_val > best_f:
                best_f = f_val
                best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END
