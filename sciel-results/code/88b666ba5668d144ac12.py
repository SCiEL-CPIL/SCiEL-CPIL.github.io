import time

def construct():
    """
    Construct a graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY single edge increases the diameter above 2
    (or disconnects the graph). Maximize edge count.
    
    The optimal structure is the complete bipartite graph K_{8,8}.
    - Vertices are split into two sets A={0..7} and B={8..15}.
    - Edges exist between every u in A and v in B.
    - Total edges: 8 * 8 = 64.
    - Diameter is 2.
    - Every edge is essential (graph is triangle-free).
    """
    deadline = time.time() + 50.0
    
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    
    # Generate all pairs between A and B
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # Safety guard (not strictly necessary for this small graph but required by directive)
    if time.time() > deadline:
        pass
        
    return edges
