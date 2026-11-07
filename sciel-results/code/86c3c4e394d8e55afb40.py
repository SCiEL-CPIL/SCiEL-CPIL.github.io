def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on 16 vertices.
    This graph has 64 edges.
    1. It is triangle-free, which is necessary for every edge to be essential 
       (if an edge were in a triangle, the alternative path of length 2 would 
       make it non-essential).
    2. By Mantel's Theorem, 64 is the maximum number of edges in a triangle-free 
       graph on 16 vertices.
    3. K_{8,8} has diameter 2. Removing any edge (u,v) leaves no path of length 2 
       between u and v (as they are in different partitions and have no common 
       neighbors), so the distance becomes 3, increasing the diameter.
    Thus, K_{8,8} is the unique optimal solution.
    """
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
