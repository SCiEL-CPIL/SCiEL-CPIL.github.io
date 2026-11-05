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

def _is_valid_graph_fast(adj_masks, n):
    """
    Check if graph has no regular induced subgraph on 6+ vertices.
    Returns True if valid, False otherwise.
    Uses bitmask operations for speed.
    """
    # We need to check all subsets S of {0..n-1} with |S| >= 6.
    # A subset S is regular if deg_S(v) is constant for all v in S.
    
    # Optimization:
    # If a subset S is regular, then for any v in S, deg_S(v) = d.
    # The sum of degrees is |S| * d, which must be even.
    
    # Strategy:
    # Iterate through all masks.
    # To speed up, we can precompute popcounts? Or just use bit_count().
    # For n=20, 2^20 = 1,048,576 masks.
    # For each mask, we iterate through set bits.
    # Average set bits is n/2 = 10.
    # So ~10M operations. This is fast in Python if using bit_count().
    
    # We can prune: if we find a regular subset, return False immediately.
    
    # To make it even faster, we can check smaller sizes first?
    # No, the probability of a small regular subset might be different.
    # But usually, random graphs don't have small regular induced subgraphs.
    # Let's just iterate all masks.
    
    # We can optimize the inner loop:
    # Pick first vertex v0.
    # d = deg(v0, S)
    # Check if all other v in S have deg(v, S) == d.
    
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
        # If not, it can't be regular.
        if (k * first_deg) % 2 != 0:
            continue
            
        is_regular = True
        for v in verts[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                is_regular = False
                break
                
        if is_regular:
            return False
            
    return True

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
        
        # Random restarts
        # We can also use local search to improve a random graph.
        # But first, let's try pure random.
        max_restarts = 100
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            # p=0.5 is standard
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            # Check validity
            if _is_valid_graph_fast(adj_masks, n):
                best_n = n
                best_adj_masks = adj_masks
                found = True
                break
                
        if found:
            break
            
    if best_n == 0:
        # If we didn't find a graph for 20-22, try smaller n
        # Or use local search on n=20?
        # Let's try local search for n=20 if random failed.
        # Actually, random graphs with n=20 are very likely to be valid.
        # If we fail, something is wrong or we need more restarts.
        
        # Let's implement a simple local search for n=20 as a fallback
        # or just try n=19, 18, etc.
        
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
                if _is_valid_graph_fast(adj_masks, n):
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
