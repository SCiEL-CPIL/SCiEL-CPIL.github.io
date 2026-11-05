import time
import random
import numpy as np

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return edges

def _find_violation(adj_masks, n):
    """
    Finds a regular induced subgraph of size >= 6.
    Returns the mask of the violation, or 0 if none found.
    """
    # Check sizes from 6 to n
    # We can generate masks of specific sizes efficiently?
    # Or just iterate all masks and filter by bit_count?
    # For n=22, 4M masks. bit_count is fast.
    
    # Optimization: 
    # If we find a violation, return it immediately.
    
    # We can also check larger sizes first? 
    # No, smaller sizes are more likely to be regular in random graphs? 
    # Actually, regularity becomes less likely as size grows? 
    # Degrees fluctuate more. 
    # But the number of subsets grows.
    # Let's just iterate all masks.
    
    # To speed up, we can precompute popcount for all masks? 
    # 2^22 is 4M integers. 4M * 8 bytes = 32MB. Acceptable.
    # But constructing it takes time. 
    # bit_count() is a C-level call, very fast.
    
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
            
        # Get first vertex
        lsb = mask & -mask
        v0 = lsb.bit_length() - 1
        d0 = (adj_masks[v0] & mask).bit_count()
        
        # Check if k * d0 is even? 
        # Sum of degrees = k * d0. Must be even.
        if (k * d0) % 2 != 0:
            continue
            
        # Check other vertices
        is_regular = True
        m = mask
        # Skip v0
        m ^= lsb
        while m:
            lsb_v = m & -m
            v = lsb_v.bit_length() - 1
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
            m ^= lsb_v
            
        if is_regular:
            return mask
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_adj_masks = []
    
    # Try n=22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        n = target_n
        found = False
        
        # Local Search
        max_iters = 5000
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            if i == 0:
                adj_masks = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)
            else:
                # Perturb previous graph
                # Pick a random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if (adj_masks[u] >> v) & 1:
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                    else:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            # Check validity
            viol_mask = _find_violation(adj_masks, n)
            if viol_mask == 0:
                best_n = n
                best_adj_masks = adj_masks
                found = True
                break
            else:
                # If violation found, we can bias the next flip to be inside the violation
                # to break the regularity.
                # For simplicity, we just do random flips in the loop above.
                # But we can be smarter: if we have a violation, flip an edge 
                # incident to a vertex in the violation.
                # Let's modify the loop to use this info.
                # Actually, the loop above is pure random walk.
                # Let's add a "greedy" step: if violation exists, flip edge in violation.
                pass
    if best_n == 0:
        # Fallback to smaller n with random restarts if local search failed
        for target_n in range(19, 4, -1):
            if time.time() > deadline - 2:
                break
            n = target_n
            for r in range(50):
                if time.time() > deadline - 2:
                    break
                adj_masks = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)
                if _find_violation(adj_masks, n) == 0:
                    best_n = n
                    best_adj_masks = adj_masks
                    break
            if best_n > 0:
                break

    if best_n == 0:
        best_n = 5
        best_adj_masks = [0] * 5

    edges = _edges_from_adj_masks(best_adj_masks, best_n)
    return {'n': best_n, 'edges': edges}
