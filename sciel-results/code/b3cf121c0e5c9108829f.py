"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    # The optimal graph is the complete bipartite graph K_{n/2, n/2}
    # This graph is triangle-free, has diameter 2, and every edge is essential
    # because removing any edge (u,v) increases the distance between u and v to 3.
    # Any graph with more edges would contain a triangle, and edges in triangles
    # are not essential for maintaining diameter 2 (removing them leaves a path of length 2).
    
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
