def construct():
    # Maximum edges in a triangle-free graph on 16 vertices is 64 (Mantel's theorem).
    # K_{8,8} achieves this bound, has diameter 2, and every edge is essential:
    # removing any edge disconnects its two endpoints (diameter becomes infinity).
    # Note: In K_{8,8}, removing (u,v) makes dist(u,v)=3, so diameter becomes 3.
    
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    result_edges = []
    for u in set_a:
        for v in set_b:
            result_edges.append([u, v])
    
    return result_edges
