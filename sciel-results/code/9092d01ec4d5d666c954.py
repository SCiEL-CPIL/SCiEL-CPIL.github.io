import time

def construct():
    """
    Constructs a C5-free graph with independence number < 9.
    Uses 8 disjoint K4s plus 3 additional vertices connected to specific K4s.
    n = 35.
    """
    n = 35
    edges = []
    
    # 1. Create 8 disjoint K4s
    # K4 i has vertices 4*i, 4*i+1, 4*i+2, 4*i+3
    for i in range(8):
        base = 4 * i
        # All pairs in K4
        for u in range(4):
            for v in range(u + 1, 4):
                edges.append([base + u, base + v])
                
    # 2. Add vertex 32 (v32)
    # Connected to all 4 vertices of K4_0
    for u in range(4):
        edges.append([32, u])
    # Connected to first vertex of K4_1 ... K4_7
    for i in range(1, 8):
        edges.append([32, 4 * i])
        
    # 3. Add vertex 33 (v33)
    # Connected to all 4 vertices of K4_1
    for u in range(4):
        edges.append([33, 4 * 1 + u])
    # Connected to first vertex of K4_0, K4_2 ... K4_7
    edges.append([33, 0]) # K4_0
    for i in range(2, 8):
        edges.append([33, 4 * i])
        
    # 4. Add vertex 34 (v34)
    # Connected to all 4 vertices of K4_2
    for u in range(4):
        edges.append([34, 4 * 2 + u])
    # Connected to first vertex of K4_0, K4_1, K4_3 ... K4_7
    edges.append([34, 0]) # K4_0
    edges.append([34, 4]) # K4_1
    for i in range(3, 8):
        edges.append([34, 4 * i])

    return {'n': n, 'edges': edges}
