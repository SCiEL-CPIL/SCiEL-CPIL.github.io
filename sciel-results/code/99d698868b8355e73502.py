import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adjacency.sum(axis=1).astype(int)
    
    # Compute core numbers
    # We maintain a list of current degrees
    curr_deg = degrees.copy()
    core_numbers = np.zeros(n, dtype=int)
    
    # To efficiently find min degree, we can use a simple loop for n=200
    # Or use a heap. For n=200, a simple linear scan or sorted list is fine.
    # Let's use a simple iterative removal.
    
    # We need to be careful: the standard core decomposition removes vertices one by one.
    # The core number of a vertex is the minimum value of the maximum degree in the subgraph 
    # containing it, as vertices are removed.
    
    # Efficient Core Number Algorithm (Batagelj & Zaversnik):
    # 1. Sort vertices by degree.
    # 2. Process vertices in order of increasing degree.
    #    When processing v, its core number is its current degree.
    #    Then, for each neighbor u of v, if curr_deg[u] > curr_deg[v], 
    #    decrement curr_deg[u].
    
    # Let's implement this.
    # We need a way to get neighbors.
    # Since n is small, we can just loop through all vertices.
    
    # Sort indices by current degree
    # We need to dynamically update.
    
    # A simpler approach for small N:
    # Repeatedly find the vertex with the minimum current degree.
    # Assign its core number.
    # Remove it and decrement neighbors.
    
    # This is O(N^2) or O(N^3) worst case, but N=200 is tiny.
    
    remaining = np.ones(n, dtype=bool)
    curr_deg = degrees.copy()
    
    for _ in range(n):
        # Find vertex with min degree among remaining
        # If all remaining have 0 degree, they all get 0.
        # We need to find the min degree value first.
        
        # Mask degrees for remaining vertices
        rem_degrees = np.where(remaining, curr_deg, np.inf)
        min_deg = np.min(rem_degrees)
        
        # Find all vertices with this min degree
        # To make it deterministic and match standard core number definition,
        # we typically process one by one, but for scoring, 
        # assigning the same core number to all with same "layer" is fine.
        # However, the exact core number depends on the order if there are ties?
        # No, the core number is well-defined.
        # The algorithm:
        # "The core number of v is the largest k such that v belongs to the k-core."
        # The iterative removal process:
        # 1. Find min degree d_min.
        # 2. All vertices with degree d_min in the current graph have core number d_min?
        #    Not exactly. If we have a cluster of degree 2, and we remove one, the others might drop to 1.
        #    So we must remove them one by one or in batches carefully.
        
        # Let's stick to the standard:
        # While there are vertices:
        #   Find v with min degree.
        #   core[v] = degree[v]
        #   Remove v, decrement neighbors.
        
        # For efficiency, we can batch remove vertices that have degree equal to the current min,
        # BUT we must ensure that removing one doesn't lower the degree of another in the same batch 
        # below the min.
        # Actually, if u and v are both degree 2, and they are connected, removing u makes v degree 1.
        # So v's core number should be 1, not 2?
        # No. If v is in the 2-core, it must have at least 2 neighbors in the 2-core.
        # If u and v are connected and both degree 2, and they are the only ones in a component, 
        # they form a K2. The 1-core is K2. The 2-core is empty.
        # So core number is 1.
        # The algorithm "assign min degree then remove" works if we process one by one.
        
        # Let's just do one by one for correctness and simplicity, N=200 is fast.
        
        min_idx = np.argmin(rem_degrees)
        core_numbers[min_idx] = curr_deg[min_idx]
        
        # Remove min_idx
        remaining[min_idx] = False
        
        # Update neighbors
        neighbors = np.where(adjacency[min_idx, :])[0]
        for u in neighbors:
            if remaining[u]:
                curr_deg[u] -= 1
                
    # Score: Lower core number is better.
    # Tie break: Lower degree is better.
    # We want higher score to be picked first.
    # So Score = - (core_number * large_factor + original_degree)
    
    # Using original degree for tie-breaking is stable.
    scores = -(core_numbers * 1000 + degrees)
    
    return scores.astype(float)
