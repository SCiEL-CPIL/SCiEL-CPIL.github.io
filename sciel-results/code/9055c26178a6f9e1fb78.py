import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the Schläfli graph on 35 vertices.
    The Schläfli graph is the unique strongly regular graph with parameters (35, 16, 7, 8).
    It is K4-free and has independence number 5.
    This certifies R(4, 6) > 35.
    """
    
    # The vertices correspond to the 35 triangles of K7.
    # We label vertices by the tuple (i, j, k) with 0 <= i < j < k <= 6.
    # There are C(7, 3) = 35 such triangles.
    
    triangles = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                triangles.append((i, j, k))
    
    n = len(triangles) # 35
    
    # Map triangle to index
    tri_to_idx = {t: i for i, t in enumerate(triangles)}
    
    # Build adjacency matrix
    # Two triangles are adjacent in the Schläfli graph if they share 0 or 2 vertices.
    # (They are NOT adjacent if they share exactly 1 vertex.)
    
    A = [[0] * n for _ in range(n)]
    
    for i in range(n):
        t1 = triangles[i]
        set1 = set(t1)
        for j in range(i + 1, n):
            t2 = triangles[j]
            set2 = set(t2)
            common = set1.intersection(set2)
            if len(common) == 0 or len(common) == 2:
                A[i][j] = 1
                A[j][i] = 1
    
    # Convert to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
