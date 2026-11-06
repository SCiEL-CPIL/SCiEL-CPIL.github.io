"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Disjoint union of 8 copies of K4: C5-free, alpha = 8 < 9, n = 32.
    # Constructed via adjacency matrix for a different approach family.
    n = 32
    adj = np.zeros((n, n), dtype=int)
    
    for block in range(8):
        start = block * 4
        end = start + 4
        adj[start:end, start:end] = 1
        np.fill_diagonal(adj[start:end, start:end], 0)
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
