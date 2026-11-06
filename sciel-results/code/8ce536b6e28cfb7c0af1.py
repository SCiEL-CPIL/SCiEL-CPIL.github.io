import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = adjacency.shape[0]
    # Ensure float for operations
    A = adjacency.astype(float)
    
    # 1. Degrees
    deg = A.sum(axis=1)
    
    # 2. Neighborhood Density / Edge Count
    # E_N(v) = number of edges between neighbors of v
    # We want to maximize this relative to possible edges.
    # Density C_v = 2 * E_N(v) / (d_v * (d_v - 1))
    
    C = np.zeros(n)
    
    # Loop over vertices to compute neighborhood edges
    # This is O(N * D^2). For N=200, D~10-20, this is 200 * 400 = 80,000 ops. Very fast.
    for v in range(n):
        if deg[v] < 2:
            C[v] = 1.0 if deg[v] == 1 else 0.0 # Isolated is 0 conflicts, Leaf is 1 conflict (dense)
            # Actually, for d=0, C is irrelevant, score is high.
            # For d=1, the neighbor set has 1 node. It's a "clique". C=1.
            continue
            
        # Get neighbors
        neighbors = np.where(A[v] > 0)[0]
        # Count edges among neighbors
        # Submatrix A[neighbors, neighbors]
        # Sum of upper triangle (to avoid double counting and self loops)
        # Since A is symmetric, we can just sum the whole thing and divide by 2
        # But we must exclude self-loops if any (graph is simple, so 0)
        sub = A[np.ix_(neighbors, neighbors)]
        edge_count = np.sum(sub) / 2.0
        
        max_edges = deg[v] * (deg[v] - 1) / 2.0
        if max_edges > 0:
            C[v] = edge_count / max_edges
        else:
            C[v] = 0.0
            
        # Time check
        if time.time() > deadline:
            break
            
    # 3. Calculate Scores
    # Score = -degree + lambda * C
    # We want low degree (negative term) and high density (positive term)
    # Let's tune lambda. 
    # Degree range: 0 to N-1.
    # C range: 0 to 1.
    # If lambda is too small, it's just LDF.
    # If lambda is large, it might override degree.
    # A vertex with d=10, C=1 should be better than d=9, C=0?
    # Score(10,1) = -10 + lambda
    # Score(9,0) = -9
    # -10 + lambda > -9 => lambda > 1.
    # So if lambda > 1, a dense degree 10 vertex beats a sparse degree 9 vertex.
    # Is this good?
    # Picking d=10 vertex kills 10 neighbors. But they are a clique, so we only "lose" 1 potential pick.
    # Picking d=9 vertex kills 9 neighbors. They are independent, so we "lose" 9 potential picks.
    # So YES, we prefer the dense one.
    # Let's set lambda = 1.5 to be safe.
    
    if time.time() > deadline:
         return -deg.astype(float)
         
    lambda_val = 1.5
    scores = -deg.astype(float) + lambda_val * C
    
    return scores
