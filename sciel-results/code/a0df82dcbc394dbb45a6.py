import time

def construct():
    # The graph is constructed as K_{3,13} with a clique on the part of size 13.
    # This graph has 117 edges and is Q3-free.
    # Vertices 0, 1, 2 form the independent set A (size 3).
    # Vertices 3..15 form the clique B (size 13).
    # Edges are all (a, b) for a in A, b in B, and all (b1, b2) for b1, b2 in B.
    
    edges = []
    A = [0, 1, 2]
    B = list(range(3, 16))
    
    # Add edges between A and B
    for a in A:
        for b in B:
            edges.append([a, b])
            
    # Add edges within B (clique)
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            edges.append([B[i], B[j]])
            
    return edges
