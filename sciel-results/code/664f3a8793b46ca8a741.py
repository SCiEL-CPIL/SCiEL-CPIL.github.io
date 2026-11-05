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

def _find_one_regular_subgraph_full(adj, n):
    """Find any regular induced subgraph of size >= 6 (exhaustive)."""
    # Check sizes in increasing order
    for size in range(6, n + 1):
        masks = _get_masks(n, size)
        for mask in masks:
            if _is_regular_induced_fast(adj, mask):
                return mask
    return None

def _find_one_regular_subgraph_sampled(adj, n, num_samples=100):
    """Find a regular induced subgraph of size 6 using sampling."""
    # We focus on size 6 as it's the most common violation
    # If we can't find a size 6 violation, we might have larger ones,
    # but breaking size 6 is the main hurdle.
    # We return a mask if found, else None.
    # Note: This is not exhaustive, so it may miss violations.
    
    # To make it more robust, we can also sample from size 7 occasionally?
    # For now, stick to size 6 for speed.
    
    # Get all 6-masks to sample from
    all_masks_6 = _get_masks(n, 6)
    
    for _ in range(num_samples):
        mask = random.choice(all_masks_6)
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
            # We use a two-phase approach:
            # 1. Sampled search to find violations quickly.
            # 2. If sampled search fails to find violations for a while, 
            #    do a full check to verify validity.
            
            found_valid = False
            steps_since_full_check = 0
            max_steps_between_full_checks = 50
            
            # Run local search for a fixed number of steps or until time limit
            max_steps = 1000
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                # Periodically do a full check if sampled check is clean
                # This ensures we don't miss larger regular subgraphs
                if steps_since_full_check >= max_steps_between_full_checks:
                    mask = _find_one_regular_subgraph_full(adj, n)
                    steps_since_full_check = 0
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
                        # Found a violation, flip an edge
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
                else:
                    # Sampled search
                    mask = _find_one_regular_subgraph_sampled(adj, n, num_samples=20)
                    steps_since_full_check += 1
                    
                    if mask is not None:
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
                    else:
                        # No violation found in sample, do full check sooner?
                        # Or just continue sampling.
                        pass

            if found_valid:
                break
                
    # Fallback if no graph > 5 found
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
