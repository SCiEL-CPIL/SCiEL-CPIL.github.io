import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Brute force over all 2^n colorings.
    """
    for mask in range(1 << n):
        valid = True
        for e in edges:
            # Check if monochromatic
            # Extract bits for vertices in e
            # e is a tuple of 5 vertices
            c0 = (mask >> e[0]) & 1
            if c0 == ((mask >> e[1]) & 1) and c0 == ((mask >> e[2]) & 1) and c0 == ((mask >> e[3]) & 1) and c0 == ((mask >> e[4]) & 1):
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    """
    Search for a small non-2-colourable 5-uniform hypergraph.
    """
    import time
    start_time = time.time()
    
    # Try n=10, then n=11
    for n in [10, 11]:
        all_edges = list(combinations(range(n), 5))
        
        # We want to find a subset of edges that is non-2-colorable.
        # Randomly permute edges and try increasing prefixes.
        random.shuffle(all_edges)
        
        # Try k edges
        # We hope to find one with k < 126
        # Start with small k, increase
        # To save time, we can binary search or just linear scan
        # Linear scan is fine if checker is fast.
        
        for k in range(40, 125):
            if time.time() - start_time > 100:
                break
                
            sample_edges = all_edges[:k]
            
            if not is_2_colorable_fast(n, sample_edges):
                return {'n': n, 'edges': [list(e) for e in sample_edges]}
                
    # Fallback: Complete 5-uniform hypergraph on 9 vertices
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
