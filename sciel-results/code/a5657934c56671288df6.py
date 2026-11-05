"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_MASKS_CACHE = {}

def _get_masks(n, size):
    """Precompute masks for subsets of a specific size."""
    key = (n, size)
    if key not in _MASKS_CACHE:
        masks = []
        for combo in combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks.append(mask)
        _MASKS_CACHE[key] = masks
    return _MASKS_CACHE[key]

def _is_regular_induced_fast(adj, mask):
    """Check if induced subgraph on mask is regular."""
    count = mask.bit_count()
    if count < 6:
        return False
    m = mask
    first_deg = -1
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        deg = (adj[idx] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _find_violation(adj, n):
    """
    Find a regular induced subgraph of size >= 6.
    Checks sizes in increasing order.
    Returns the mask of the violation, or None if valid.
    """
    for size in range(6, n + 1):
        masks = _get_masks(n, size)
        for mask in masks:
            if _is_regular_induced_fast(adj, mask):
                return mask
    return None

def _extract_vertices(mask):
    """Extract list of vertex indices from mask."""
    verts = []
    m = mask
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        verts.append(idx)
        m ^= lb
    return verts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n=22 down to 15
    # We prioritize larger n
    for target_n in range(22, 14, -1):
        if time.time() > deadline - 5:
            break
            
        n = target_n
        
        # Number of restarts depends on n
        # We want to try harder for larger n
        max_restarts = 10 if n >= 21 else 20
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search
            found_valid = False
            max_steps = 2000
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                # Find a violation
                mask = _find_violation(adj, n)
                
                if mask is None:
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found_valid = True
                    break
                else:
                    # Found a violation, flip an edge inside it
                    verts = _extract_vertices(mask)
                    if len(verts) >= 2:
                        u_idx, v_idx = random.sample(range(len(verts)), 2)
                        u = verts[u_idx]
                        v = verts[v_idx]
                        if (adj[u] >> v) & 1:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

            if found_valid:
                break
                
    # Fallback if no graph > 5 found
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
