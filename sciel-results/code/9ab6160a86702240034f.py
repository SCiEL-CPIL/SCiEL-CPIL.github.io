import numpy as np
import time

def construct():
    n = 40
    m = 401
    deadline = time.time() + 110  # Safety margin
    
    best_edges = None
    best_score = float('inf')
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_edges = len(all_edges)
    
    # Initial random guess
    import random
    random.seed(42) # For reproducibility in testing, though evaluator might not care
    
    while time.time() < deadline:
        # Generate a random graph
        edges = random.sample(all_edges, m)
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        # Calculate common neighbors for all pairs
        # common[u, v] = number of common neighbors of u and v
        common = A @ A
        
        # Find max book size among edges
        # We only care about pairs (u, v) where A[u, v] == 1
        # Create a mask for edges
        # Since A is symmetric and 0 diagonal, A > 0 is the edge mask
        edge_mask = A > 0
        # common is symmetric too
        # We can just take max of common[edge_mask]
        # But common[u, v] for u!=v is the book size.
        # Diagonal is degree, not relevant for edges.
        
        # To be safe and fast:
        # max_book = np.max(common[edge_mask])
        # This includes both (u,v) and (v,u), which is fine.
        
        current_score = np.max(common[edge_mask])
        
        if current_score < best_score:
            best_score = current_score
            best_edges = edges
            
            # If we hit the theoretical lower bound, we can stop early?
            # The bound is 40/6 = 6.66, so integer 7.
            # If we find 7, it's likely optimal or very close.
            # Let's check if 7 is achievable.
            if best_score <= 7:
                break

    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
