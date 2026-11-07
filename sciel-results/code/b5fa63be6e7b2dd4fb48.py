import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Identify edge types
    bottom_edges = []  # edges in bottom cycle (indices in 'edges' list)
    top_edges = []     # edges in top cycle
    match_edges = []   # matching edges
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)
            
    # We fix bottom and matching edges to +1.
    # We vary top edges.
    # The lift is connected iff at least one top edge is -1.
    
    # Pre-compute the fixed part of the adjacency matrix
    A_fixed = np.zeros((n, n), dtype=np.float64)
    for i in bottom_edges:
        u, v = edges[i]
        A_fixed[u, v] = 1.0
        A_fixed[v, u] = 1.0
    for i in match_edges:
        u, v = edges[i]
        A_fixed[u, v] = 1.0
        A_fixed[v, u] = 1.0
        
    best_signs = None
    best_radius = np.inf
    
    # Enumerate all 2^m sign assignments for top edges
    # m is small (4, 5, 7, 9)
    num_top = len(top_edges)
    
    for mask in range(1 << num_top):
        if time.time() > deadline:
            break
            
        # Skip mask 0 (all +1) because it leads to disconnected lift
        if mask == 0:
            continue
            
        # Construct signs array
        signs = np.ones(E, dtype=np.int8)
        # Bottom and matching are already +1 (default)
        
        # Set top signs based on mask
        for j in range(num_top):
            if (mask >> j) & 1:
                signs[top_edges[j]] = -1
            else:
                signs[top_edges[j]] = 1
                
        # Construct full adjacency matrix
        A = A_fixed.copy()
        for j in range(num_top):
            i = top_edges[j]
            u, v = edges[i]
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        try:
            eigvals = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigvals))
        except:
            continue
            
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # If no valid signs found (should not happen for m>=1), fallback
    if best_signs is None:
        # Fallback to a random valid sign
        # Ensure at least one -1 in top
        best_signs = np.ones(E, dtype=np.int8)
        if top_edges:
            best_signs[top_edges[0]] = -1
        # Verify validity with check_lift just in case
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            check_lift(graph, best_signs)
        except:
            # If even this fails, try another
            for i in range(E):
                best_signs[i] = -1
                try:
                    check_lift(graph, best_signs)
                    break
                except:
                    best_signs[i] = 1

    return best_signs
