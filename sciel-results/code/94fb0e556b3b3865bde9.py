def construct():
    """
    Constructs a maximal edge-critical diameter-2 graph on 16 vertices.
    
    The optimal graph is the complete bipartite graph K_{8,8}.
    - Vertices are partitioned into two sets A and B of size 8 each.
    - All edges connect a vertex in A to a vertex in B.
    - Total edges: 8 * 8 = 64.
    - Diameter is 2.
    - Edge-critical: Removing any edge (u, v) where u in A and v in B 
      leaves u and v with no common neighbors (since A and B are independent sets 
      and all cross-edges are present except the removed one, but wait: 
      in K_{n,n}, u has neighbors B\{v} and v has neighbors A\{u}. 
      Common neighbors of u and v in G-e: 
      A vertex w is a common neighbor if w is adjacent to both u and v.
      If w in A, w-v is an edge, but w-u is not (u,w in A).
      If w in B, w-u is an edge, but w-v is not (v,w in B).
      So no common neighbors. Distance becomes 3. Diameter increases to 3.
    """
    n = 16
    half = n // 2
    
    # Partition vertices
    set_a = list(range(half))
    set_b = list(range(half, n))
    
    edges = []
    
    # Generate all edges between A and B
    for u in set_a:
        for v in set_b:
            edges.append([int(u), int(v)])
            
    return edges
