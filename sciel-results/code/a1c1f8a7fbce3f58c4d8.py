import time

def construct():
    # Generate all 16 even-weight binary strings of length 5
    # These form the vertex set of the Clebsch graph
    vertices = []
    for i in range(32):
        # Check if weight is even
        w = bin(i).count('1')
        if w % 2 == 0:
            vertices.append(i)
            
    # Map vertices to 0..15
    # vertices list contains 16 integers from 0..31
    # We need to ensure we have exactly 16
    assert len(vertices) == 16
    
    # Create mapping from vertex value to index 0..15
    # The order doesn't matter for isomorphism, but we need consistent IDs
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    
    # Connect vertices if Hamming distance is 4
    # In the Clebsch graph (folded 5-cube restricted to even weights),
    # adjacency is defined by distance 4 (which corresponds to distance 1 in the folded graph)
    for i in range(16):
        for j in range(i + 1, 16):
            u = vertices[i]
            v = vertices[j]
            diff = u ^ v
            if bin(diff).count('1') == 4:
                edges.append([i, j])
                
    return edges
