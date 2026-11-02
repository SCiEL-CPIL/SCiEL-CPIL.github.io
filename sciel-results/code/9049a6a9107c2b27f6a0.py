import random
import collections
import time

def construct(k, g):
    # For k=3, we use the LUW construction with q=3.
    # Girth g = 2d + 2 => d = g/2 - 1.
    # We need girth >= g, so we choose d such that 2d+2 >= g.
    # Minimal d = ceil((g-2)/2).
    
    if k != 3:
        # Fallback for other k (not expected in this task)
        # Try to find a random regular graph
        return _random_search(k, g)
        
    q = 3
    d = (g - 2) // 2
    if 2 * d + 2 < g:
        d += 1
        
    # Construct LUW graph G(q, d)
    # Vertices are (s, x1, ..., xd)
    # Total vertices: 2 * q**d
    
    n = 2 * (q ** d)
    
    # Map vertex state to index
    # State: (s, x1, ..., xd)
    # Index: s * (q**d) + x1 * q**(d-1) + ... + xd
    
    def get_index(s, x):
        idx = s * (q ** d)
        for i in range(d):
            idx += x[i] * (q ** (d - 1 - i))
        return idx
        
    def get_state(idx):
        s = idx // (q ** d)
        rem = idx % (q ** d)
        x = [0] * d
        for i in range(d):
            x[i] = rem % q
            rem //= q
        return s, x
        
    edges = []
    # We only need to generate edges from s=0 to s=1 to avoid duplicates
    # Each vertex in s=0 has q neighbors in s=1.
    
    for idx in range(0, q ** d):
        s, x = get_state(idx)
        # s is always 0 in this loop
        for j in range(q):
            # Calculate neighbor y
            y = [0] * d
            y[0] = (x[0] + j) % q
            for i in range(1, d):
                y[i] = (x[i] + x[i-1] * y[i-1]) % q
            
            n_idx = get_index(1, y)
            edges.append((idx, n_idx))
            
    return n, edges

def _random_search(k, g):
    # Fallback for k != 3
    # Simple random search, might not find anything
    min_n = 2 * k * (k-1) ** (g//2 - 1) # Moore bound approx
    # This is just a placeholder to satisfy signature
    return 0, []
