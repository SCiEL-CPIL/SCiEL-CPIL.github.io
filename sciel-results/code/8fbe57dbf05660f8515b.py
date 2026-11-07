"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the edge list of K_{8,8}, the complete bipartite graph on 16 vertices
    partitioned into two sets of 8.
    
    This graph has:
    - 64 edges (maximal for a triangle-free graph on 16 vertices by Mantel's theorem)
    - Diameter exactly 2 (any two vertices in different parts are adjacent,
      any two in the same part have all vertices in the other part as common neighbors)
    - Every edge is essential: removing edge (u,v) where u in part A, v in part B
      eliminates the only length-1 path between u and v. Since the graph is
      triangle-free, u and v have no common neighbors, so there is no length-2 path
      between them either. Thus dist(u,v) becomes 3 (via u -> any other vertex in A -> ... 
      wait, actually u is in A, v is in B. After removing (u,v), to go from u to v,
      u must go to some vertex in B (other than v), then from there to v? No, vertices
      in B are not connected to each other. So u -> w (in B, w != v) -> ... but w and v 
      are both in B, no edge between them. So we need u -> w (in B) -> x (in A) -> v.
      That's length 3. So diameter becomes 3.)
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
