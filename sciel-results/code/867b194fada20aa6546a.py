import time
import random
import numpy as np
from itertools import combinations

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        # Only consider j > i to avoid duplicates
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return edges

def _find_first_regular_subgraph(adj_masks, n):
    """
    Returns the bitmask of the first regular induced subgraph of size >= 6 found.
    Returns None if no such subgraph exists (graph is valid).
    """
    # Iterate through all masks with popcount >= 6
    # To speed up, we can iterate by size or just all masks.
    # Iterating all masks 1..2^n is simple.
    
    # Optimization: Precompute list of vertices for each mask? No, too much memory.
    # Just extract on the fly.
    
    # We can randomize the order of masks to find a "bad" subgraph faster?
    # Or just iterate sequentially.
    # Sequential is fine.
    
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
            
        first_deg = (adj_masks[verts[0]] & mask).bit_count()
        
        # Quick check: is k * first_deg even?
        if (k * first_deg) % 2 != 0:
            continue
            
        is_regular = True
        for v in verts[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                is_regular = False
                break
                
        if is_regular:
            return mask
            
    return None

def _random_flip(adj_masks, n):
    """Flip a random edge in the graph."""
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    if i == j:
        return
    if i > j:
        i, j = j, i
    # Flip edge (i, j)
    bit = 1 << j
    adj_masks[i] ^= bit
    bit_i = 1 << i
    adj_masks[j] ^= bit_i

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 0
    best_adj_masks = []
    
    # Try n=22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        n = target_n
        found = False
        
        # Random restarts with local search
        max_restarts = 100
        max_steps_per_restart = 50
        
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
                        
            # Local Search
            for step in range(max_steps_per_restart):
                if time.time() > deadline - 5:
                    break
                    
                reg_mask = _find_first_regular_subgraph(adj_masks, n)
                
                if reg_mask is None:
                    # Valid graph found
                    best_n = n
                    best_adj_masks = adj_masks
                    found = True
                    break
                else:
                    # Flip a random edge
                    _random_flip(adj_masks, n)
                    
            if found:
                break
                
        if found:
            break
            
    if best_n == 0:
        # Fallback to smaller n if 20-22 failed
        for target_n in range(19, 4, -1):
            if time.time() > deadline - 2:
                break
            n = target_n
            for r in range(50):
                if time.time() > deadline - 2:
                    break
                adj_masks = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                if _find_first_regular_subgraph(adj_masks, n) is None:
                    best_n = n
                    best_adj_masks = adj_masks
                    break
            if best_n > 0:
                break
                
    if best_n == 0:
        # Absolute fallback: n=5
        best_n = 5
        best_adj_masks = [0] * 5

    edges = _edges_from_adj_masks(best_adj_masks, best_n)
    return {'n': best_n, 'edges': edges}
