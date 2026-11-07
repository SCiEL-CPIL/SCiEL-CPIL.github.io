import time

P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    edges = []
    
    # Construct K_{half, half}
    # This graph is triangle-free, has diameter 2, and every edge is essential.
    # Removing any edge (u, v) increases the distance between u and v to 3,
    # thus increasing the diameter of the graph to 3.
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    return edges
