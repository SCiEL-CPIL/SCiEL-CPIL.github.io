import numpy as np

def priority(adjacency):
    adj = np.asarray(adjacency, dtype=int)
    n = adj.shape[0]
    
    # Compute Core Numbers
    # 1. Calculate degrees
    degrees = adj.sum(axis=1)
    
    # 2. Sort vertices by degree
    order = np.argsort(degrees)
    
    # 3. Peeling process
    # We need to simulate removing vertices with min degree
    # For n=200, we can do this simply
    remaining = np.ones(n, dtype=bool)
    current_degrees = degrees.copy()
    core_numbers = np.zeros(n, dtype=int)
    
    # Bucket sort approach for efficiency, but n is small so simple loop works
    # However, standard core number algorithm:
    # Sort by degree.
    # Iterate through sorted list.
    # For each vertex v in sorted order:
    #   core[v] = max(core[v], current_degrees[v])
    #   For each neighbor u of v that is still remaining:
    #       if current_degrees[u] > core[v]:
    #           current_degrees[u] -= 1
    #           # This requires re-sorting or a priority queue for strict correctness
    #           # But for small n, we can just use a priority queue or repeated min-finding.
    
    # Simpler O(n^2) approach for n=200:
    # 1. Create list of vertices
    # 2. While list not empty:
    #    a. Find vertex with min degree in remaining graph
    #    b. Assign core number = current min degree
    #    c. Remove vertex and update degrees of neighbors
    
    vertices = list(range(n))
    remaining_degrees = degrees.copy()
    
    while vertices:
        # Find min degree vertex
        min_idx = 0
        min_deg = remaining_degrees[vertices[0]]
        for i in range(1, len(vertices)):
            if remaining_degrees[vertices[i]] < min_deg:
                min_deg = remaining_degrees[vertices[i]]
                min_idx = i
        
        v = vertices[min_idx]
        core_numbers[v] = min_deg
        
        # Remove v
        vertices.pop(min_idx)
        
        # Update degrees of neighbors
        neighbors = np.where(adj[v] == 1)[0]
        for u in neighbors:
            if u in vertices:
                # This check is O(n), making total O(n^3) worst case.
                # n=200, 200^3 = 8,000,000 ops. This is fine for 120s limit.
                remaining_degrees[u] -= 1
                
    # Heuristic: 
    # Low core number means the vertex is in a sparse part of the graph.
    # We want to pick these first? 
    # Or do we want to pick high core number vertices?
    # Intuition: Dense cores are "hard". Sparse parts are "easy".
    # If we pick sparse parts first, we don't interfere with the dense core structure as much?
    # Actually, usually you want to pick low degree vertices. Core number correlates with degree.
    # Let's try Score = -CoreNumber.
    
    return -core_numbers
