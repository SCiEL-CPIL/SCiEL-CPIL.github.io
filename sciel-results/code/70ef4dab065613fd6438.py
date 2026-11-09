"""
Search for a large graph with no regular induced subgraph on 6+ vertices.
Uses bitmask representation and vectorized numpy checks for speed.
"""
import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid graph for n=20 first
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        # Try random graphs with various seeds
        found = False
        for seed in range(500):
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(seed)
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if _is_valid(adj, target_n):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
        
        if found:
            break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: trivially valid small graph
    return {'n': 5, 'edges': []}


def _is_valid(adj, n):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Uses bitmask representation and vectorized numpy operations.
    """
    # Precompute all 6-subsets as bitmasks for faster checking
    # For n up to 22, C(22,6) = 74613, which is manageable
    
    # Check all subsets of size 6
    # A subset of size k is regular if all vertices in it have the same degree within the subset
    
    # For speed, we check size 6 first (most common violations)
    # If any size-6 subset is regular, return False
    
    # Use itertools.combinations for size 6
    import itertools
    
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj, subset, size):
                return False
    return True


def _is_regular_subset(adj, subset, size):
    """Check if induced subgraph on subset is regular."""
    # Compute degree of each vertex in subset within the induced subgraph
    degs = []
    for v in subset:
        deg = 0
        for u in subset:
            if u != v and (adj[v] & (1 << u)):
                deg += 1
        degs.append(deg)
    
    # Regular if all degrees equal
    return all(d == degs[0] for d in degs)
