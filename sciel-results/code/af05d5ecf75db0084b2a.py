def construct():
    # Two disjoint K_{3,3} components: vertices 0-5 and 6-11.
    # K_{3,3} is bipartite (triangle-free), 3-regular, and has cycle lengths {4, 6}.
    # Two disjoint copies give exactly 2 distinct cycle lengths.
    
    edges = []
    
    def add_k33(offset):
        p1 = [offset, offset + 1, offset + 2]
        p2 = [offset + 3, offset + 4, offset + 5]
        for u in p1:
            for v in p2:
                a, b = (u, v) if u < v else (v, u)
                edges.append([int(a), int(b)])
    
    add_k33(0)
    add_k33(6)
    
    edges.sort()
    return edges
