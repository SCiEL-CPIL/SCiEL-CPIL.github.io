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

def _find_first_regular_subgraph(adj_masks, n):
    """
    Finds the first regular induced subgraph of size >= 6.
    Returns (mask, vertices_list) or (None, None).
    """
    # Iterate masks
    # Optimization: 
    # We can precompute popcounts for all masks? 
    # For n=21, 2^21 is ~2M. 
    # A list of 2M small ints is ~16MB. Acceptable.
    # But doing it every call is expensive. 
    # Let's just use bit_count().
    
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
            
        if not verts:
            continue
            
        # Check regularity
        first_deg = (adj_masks[verts[0]] & mask).bit_count()
        
        # Parity check: sum of degrees = k * first_deg must be even
        if (k * first_deg) % 2 != 0:
            continue
            
        is_regular = True
        for v in verts[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                is_regular = False
                break
                
        if is_regular:
            return mask, verts
            
    return None, None

def _local_search(adj_masks, n, max_steps, deadline):
    """
    Performs local search to eliminate regular subgraphs.
    Returns (adj_masks, is_valid).
    """
    for _ in range(max_steps):
        if time.time() > deadline:
            return adj_masks, False
            
        mask, verts = _find_first_regular_subgraph(adj_masks, n)
        
        if mask is None:
            return adj_masks, True
            
        # Flip a random edge within the regular subgraph
        # Pick two distinct vertices from verts
        if len(verts) < 2:
            continue
            
        i, j = random.sample(verts, 2)
        
        # Flip edge (i, j)
        bit_j = 1 << j
        bit_i = 1 << i
        adj_masks[i] ^= bit_j
        adj_masks[j] ^= bit_i
        
    return adj_masks, False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_adj_masks = []
    
    # Prioritize higher n
    # n=22 is very hard, n=21 is hard, n=20 is easy.
    
    # Strategy:
    # 1. Try n=22 with limited effort (random + short local search)
    # 2. Try n=21 with more effort (random + longer local search)
    # 3. Try n=20 with random sampling (should succeed quickly)
    
    targets = [
        (22, 5, 10),   # n, max_restarts, max_steps_per_restart
        (21, 10, 50),
        (20, 50, 10)   # For 20, random is likely enough, but local search is fast if needed
    ]
    
    for n, max_restarts, max_steps in targets:
        if time.time() > deadline - 5:
            break
            
        found = False
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            # Check if already valid
            mask, verts = _find_first_regular_subgraph(adj_masks, n)
            if mask is None:
                best_n = n
                best_adj_masks = adj_masks
                found = True
                break
            
            # Otherwise, local search
            adj_masks, is_valid = _local_search(adj_masks, n, max_steps, deadline)
            if is_valid:
                best_n = n
                best_adj_masks = adj_masks
                found = True
                break
                
        if found:
            break
            
    # Fallback for smaller n if 20-22 failed
    if best_n < 20:
        for n in range(19, 4, -1):
            if time.time() > deadline - 2:
                break
            for r in range(20):
                if time.time() > deadline - 2:
                    break
                adj_masks = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                if _find_first_regular_subgraph(adj_masks, n)[0] is None:
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
