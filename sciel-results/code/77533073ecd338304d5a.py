import numpy as np

def construct():
    # We construct two disjoint K_{3,3} graphs.
    # Vertices 0-5 form the first K_{3,3} (partitions {0,1,2} and {3,4,5}).
    # Vertices 6-11 form the second K_{3,3} (partitions {6,7,8} and {9,10,11}).
    
    # Start with an empty 12x12 adjacency matrix
    adj = np.zeros((12, 12), dtype=int)
    
    # Helper to add edges for a K_{3,3} component
    def add_k33(offset):
        left = [offset + 0, offset + 1, offset + 2]
        right = [offset + 3, offset + 4, offset + 5]
        for u in left:
            for v in right:
                adj[u][v] = 1
                adj[v][u] = 1

    add_k33(0)
    add_k33(6)
    
    # Extract edges
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if adj[i][j] == 1:
                edges.append([int(i), int(j)])
                
    return edges
