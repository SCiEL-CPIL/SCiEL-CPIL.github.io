import time
import itertools

def construct():
    # The Clebsch graph is the optimal triangle-free 4-chromatic graph on 16 vertices.
    # It has 40 edges.
    # Construction:
    # 1. Consider the 5-dimensional hypercube Q5 (32 vertices, binary strings of length 5).
    # 2. Take the 16 vertices with even Hamming weight.
    # 3. Connect two vertices if their Hamming distance is 4.
    
    # Generate even-weight binary strings of length 5
    verts = []
    for bits in itertools.product([0, 1], repeat=5):
        if sum(bits) % 2 == 0:
            verts.append(bits)
            
    # There should be 16 such vertices
    # Map them to indices 0..15
    # The order doesn't matter for the graph structure, but we need consistent indices.
    
    edges = []
    n = 16
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Hamming distance
            dist = 0
            for k in range(5):
                if verts[i][k] != verts[j][k]:
                    dist += 1
            if dist == 4:
                edges.append([i, j])
                
    # The resulting graph has 40 edges.
    # It is triangle-free and 4-chromatic.
    # It is a valid graph on vertices 0..15.
    
    return edges
