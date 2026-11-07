"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
    
    # Precompute is_clique and is_ind_set for all masks
    # mask: integer representing set of vertices
    is_clique = [False] * (1 << n)
    is_ind_set = [False] * (1 << n)
    is_clique[0] = True
    is_ind_set[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest]:
            ok = True
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if not adj[v][u]:
                    ok = False
                    break
                r ^= lsb_r
            if ok:
                is_clique[mask] = True
        
        # Check ind_set: v must NOT be connected to any in rest, and rest must be ind_set
        if is_ind_set[rest]:
            ok = True
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if adj[v][u]:
                    ok = False
                    break
                r ^= lsb_r
            if ok:
                is_ind_set[mask] = True

    # List of all valid masks (non-empty)
    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind_set[mask]:
            valid_masks.append(mask)

    # Memoization for can_partition
    # State: (mask, k)
    # Returns: True if mask can be partitioned into <= k valid parts
    # We also need to store the choice to reconstruct
    choice = {}
    
    @lru_cache(maxsize=None)
    def can_partition(mask, k):
        if mask == 0:
            return True
        if k == 0:
            return False
        
        # Optimization: if mask itself is valid, we can use 1 part
        if is_clique[mask] or is_ind_set[mask]:
            choice[(mask, k)] = mask
            return True
        
        # Pick lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # Try all valid subsets S of mask that contain v
        for S in valid_masks:
            if (S & mask) == S and (S & lsb) != 0:
                if can_partition(mask ^ S, k - 1):
                    choice[(mask, k)] = S
                    return True
        
        return False

    # Find minimum k
    best_k = -1
    for k in range(1, n + 1):
        if can_partition(full_mask, k):
            best_k = k
            break
    
    if best_k == -1:
        # Fallback: should not happen
        return [[i] for i in range(n)]
    
    # Reconstruct partition
    parts = []
    mask = full_mask
    k = best_k
    while mask != 0:
        S = choice[(mask, k)]
        # Convert mask S to list of vertices
        part_verts = []
        m = S
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_verts.append(v)
            m ^= lsb
        parts.append(part_verts)
        mask ^= S
        k -= 1
    
    # Ensure types are correct: list of lists of ints
    return parts

# EVOLVE-BLOCK-END
