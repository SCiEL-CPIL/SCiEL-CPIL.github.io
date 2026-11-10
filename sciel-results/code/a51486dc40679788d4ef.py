"""Improved program for dense edge-critical diameter 2 graphs."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of edges representing the complete bipartite graph K_{8,8}.
    
    Analysis:
    1. The graph must have diameter exactly 2.
    2. Deleting EACH individual edge must increase the diameter above 2.
       This implies the graph is "minimally diameter 2".
    3. A graph is minimally diameter 2 if and only if it has diameter 2 and 
       contains no triangles (girth >= 5).
       - If an edge (u,v) is in a triangle (u,v,w), removing (u,v) leaves path u-w-v (len 2), 
         so diameter stays 2.
       - If no triangles exist, removing (u,v) means no path of length 2 between u and v. 
         Since original diameter was 2, d(u,v) was 1. New d(u,v) >= 3 or infinity. 
         Thus diameter increases.
    4. We need to maximize edges in a triangle-free graph on 16 vertices with diameter 2.
    5. Turan's Theorem states the max edges in a triangle-free graph on n vertices is 
       floor(n^2 / 4). For n=16, max edges = 16^2 / 4 = 64.
    6. The complete bipartite graph K_{8,8} has 64 edges, is triangle-free, and has 
       diameter 2 (nodes in same partition are dist 2, different partitions dist 1).
    7. Therefore, K_{8,8} is the unique optimal solution (up to isomorphism) with 64 edges.
    """
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
