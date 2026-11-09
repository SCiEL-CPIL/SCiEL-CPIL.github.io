def construct():
    """
    Return edges of the complete bipartite graph K_{8,8} on vertices 0..15.
    
    Properties:
    - Diameter is exactly 2.
    - Every edge is critical: deleting any edge increases the distance between its endpoints to 3, 
      thus increasing the diameter to 3.
    - Edge count: 64.
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges
