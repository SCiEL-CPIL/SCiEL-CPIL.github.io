import itertools

def construct():
    n = 47
    # The complement graph H is a matching.
    # H has edges (0,1), (2,3), ..., (44,45). Vertex 46 is isolated.
    # G is the complement of H.
    
    # Identify edges in H
    h_edges = set()
    for i in range(0, n - 1, 2):
        h_edges.add((i, i + 1))
        
    g_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in h_edges:
                g_edges.append([u, v])
                
    return {'n': n, 'edges': g_edges}
