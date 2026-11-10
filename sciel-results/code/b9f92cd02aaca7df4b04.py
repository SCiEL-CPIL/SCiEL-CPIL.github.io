import time
import random
from itertools import combinations

def is_valid_graph(n, edges):
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check all subsets of size >= 6
    # Optimization: iterate through all masks
    # If n is small, we can just loop.
    # For n=22, 2^22 is ~4M.
    
    # Early exit if invalid
    # We can check subsets of size 6 first? No, must check all.
    # But if a subset of size 6 is regular, it's invalid.
    # If a subset of size 7 is regular, it's invalid.
    
    # Let's iterate all masks
    # To speed up, we can skip masks with < 6 bits.
    
    # Note: An induced subgraph is regular if all vertices in the subset have the same degree within the subset.
    
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        # Check if regular
        # Get degrees of vertices in mask
        # Optimization: pick one vertex, get its degree, check if all others match.
        
        # Find first set bit
        # We can just iterate all set bits
        first_deg = -1
        is_reg = True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                is_reg = False
                break
            m -= lsb
        
        if is_reg:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try n from 22 down to 10
    for n in range(22, 9, -1):
        # Try several random graphs
        # Edge probability p around 0.5 is typical
        # Also try p=0.1, 0.9, etc?
        # Actually, random graphs with p=0.5 are most "generic".
        
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            if is_valid_graph(n, edges):
                return {'n': n, 'edges': edges}
                
            # If n is large, random search might be sparse in valid graphs.
            # Maybe try a local search or specific construction?
            # For now, just keep trying random.
            
            # If we've tried many, maybe break to smaller n?
            # But we have time. Let's try to be smart.
            # If n=22 fails after 100 tries, maybe it's rare.
            # But 4M subsets is not that many.
            
            # Let's add a counter to break if too many attempts for this n
            if attempts > 50 and n > 15:
                break

    # Fallback to known small valid graphs
    # n=5 empty graph
    return {'n': 5, 'edges': []}
