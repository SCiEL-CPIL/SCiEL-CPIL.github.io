def construct():
    # Construct the Clebsch graph on 16 vertices.
    # This graph is triangle-free, 5-regular, and has chromatic number 4.
    # It has 40 edges, which is optimal for this task.
    
    edges = []
    n = 16
    
    # Add edges corresponding to Hamming distance 1
    # Iterate through each vertex and flip each of the 4 bits
    for u in range(n):
        for bit_idx in range(4):
            v = u ^ (1 << bit_idx)
            # Ensure each edge is added only once
            if u < v:
                edges.append([u, v])
                
    # Add edges corresponding to Hamming distance 4 (antipodal vertices)
    for u in range(n):
        v = u ^ 15  # Flip all 4 bits
        if u < v:
            edges.append([u, v])
            
    return edges
