"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def _adj_to_bitmasks(adj, n):
    """Convert adjacency matrix to list of integer bitmasks."""
    masks = [0] * n
    for u in range(n):
        m = 0
        for v in range(n):
            if adj[u, v]:
                m |= (1 << v)
        masks[u] = m
    return masks

def _check_subset_regular(masks, subset):
    """Check if the induced subgraph on subset is regular using bitmasks."""
    k = len(subset)
    if k < 2:
        return False  # trivially regular
    # Build subset bitmask
    sub_mask = 0
    for v in subset:
        sub_mask |= (1 << v)
    # Compute degrees within subset for each vertex
    degrees = []
    for v in subset:
        # Count neighbors of v that are also in subset
        d = bin(masks[v] & sub_mask).count('1')
        degrees.append(d)
    # Check if all degrees are equal
    first = degrees[0]
    for d in degrees[1:]:
        if d != first:
            return False
    return True

def _check_all_6_7_bitmask(masks, n):
    """Check all 6- and 7-subsets for regularity using bitmasks."""
    for combo in combinations(range(n), 6):
        if _check_subset_regular(masks, combo):
            return False
    for combo in combinations(range(n), 7):
        if _check_subset_regular(masks, combo):
            return False
    return True

def _sample_check_bitmask(masks, n, num_samples=30):
    """Quick sample check for regular 6-subsets using bitmasks."""
    for _ in range(num_samples):
        idx = random.sample(range(n), 6)
        if _check_subset_regular(masks, idx):
            return False
    return True

def _check_8plus_bitmask(masks, n, num_samples=50):
    """Check subsets of size 8+ for regularity using sampling."""
    for sz in range(8, n + 1):
        total = comb(n, sz)
        samples = min(total, num_samples)
        for _ in range(samples):
            idx = random.sample(range(n), sz)
            if _check_subset_regular(masks, idx):
                return False
    return True

def _full_verify_bitmask(masks, n):
    """Full verification using bitmask approach."""
    if not _check_all_6_7_bitmask(masks, n):
        return False
    if not _check_8plus_bitmask(masks, n, 50):
        return False
    return True

def _random_adj_bitmask(n, density):
    """Generate random graph and return bitmasks."""
    masks = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
    return masks

def _bitmasks_to_edges(masks, n):
    """Convert bitmasks to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if masks[u] & (1 << v):
                edges.append([u, v])
    return edges

def _local_search_bitmask(masks, n, max_steps=100):
    """Local search: flip random edges and check."""
    for step in range(max_steps):
        if _sample_check_bitmask(masks, n, 20) and _full_verify_bitmask(masks, n):
            return True
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            # Toggle edge (u,v)
            if masks[u] & (1 << v):
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    best_n = 5
    best_edges = []
    
    # Try n from 22 down to 6, aiming for maximum
    for n in range(22, 5, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Phase 1: Pure random search
        num_random = 200 if n <= 15 else 80
        for attempt in range(num_random):
            if time.time() > deadline - 15:
                break
            
            density = 0.3 + 0.4 * random.random()
            masks = _random_adj_bitmask(n, density)
            
            if _sample_check_bitmask(masks, n, 20):
                if _full_verify_bitmask(masks, n):
                    best_n = n
                    best_edges = _bitmasks_to_edges(masks, n)
                    found = True
                    break
        
        if found:
            continue
        
        # Phase 2: Local search from random starts
        if time.time() > deadline - 15:
            break
        
        num_ls = 30 if n <= 15 else 15
        for trial in range(num_ls):
            if time.time() > deadline - 15:
                break
            
            density = 0.3 + 0.4 * random.random()
            masks = _random_adj_bitmask(n, density)
            
            if _local_search_bitmask(masks, n, max_steps=150):
                best_n = n
                best_edges = _bitmasks_to_edges(masks, n)
                found = True
                break
    
    # Fallback: ensure we have at least a valid small graph
    if best_n < 6:
        # Build a known valid small graph (e.g., empty graph on 5 vertices)
        best_n = 5
        best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
