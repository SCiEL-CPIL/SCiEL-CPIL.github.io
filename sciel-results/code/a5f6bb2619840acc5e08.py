"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 40
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    import random
    np.random.seed(42)
    
    for restart in range(50):
        np.random.seed(restart * 7 + 13)
        edge_order = list(all_edges)
        random.shuffle(edge_order)
        
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        
        for (u, v) in edge_order:
            # Check if there's a walk of length 5 from u to v
            # A^5[u,v] > 0 means such a walk exists
            # Compute A^5 efficiently by repeated squaring/multiplication
            # For small n=40, just do 5 matrix multiplications
            A5 = A.copy()
            for _ in range(4):
                A5 = A5 @ A
            
            if A5[u, v] == 0:
                # Safe to add
                A[u, v] = 1
                A[v, u] = 1
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
