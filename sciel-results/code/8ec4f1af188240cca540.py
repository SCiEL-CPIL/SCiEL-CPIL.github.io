"""Twin clique pancyclic construction on 40 vertices."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    
    # Clique A: vertices 0..19
    for i in range(20):
        for j in range(i + 1, 20):
            edges.add((i, j))
    
    # Clique B: vertices 20..39
    for i in range(20, 40):
        for j in range(i + 1, 40):
            edges.add((i, j))
    
    # Perfect matching: i <-> i+20 for i in 0..19
    for i in range(20):
        edges.add((i, i + 20))
    
    edges_list = sorted(edges)
    
    cycles = []
    for k in range(3, N + 1):
        if k <= 20:
            # Cycle entirely within clique A
            cycles.append(list(range(k)))
        elif k == 21:
            # 1 vertex from A, 20 from B: v0, b20, b21, ..., b39, v0
            # b20 = 20 (matched to v0), b21..b39 = 21..39
            # Edges: (0,20) matching, (20,21)..(38,39) clique B, (39,0) clique A
            cyc = [0] + list(range(20, 40))
            cycles.append(cyc)
        else:
            # k in 22..40: use (k-20) vertices from A and 20 from B
            a_count = k - 20  # 2..20
            # A vertices: 0, 1, ..., a_count-1
            # B vertices: 20, 21, ..., 39
            # Pattern: a0, b20, a1, b21, a2, b22, ..., a_{a-1}, b_{20+a-1}, b_{21+a}, ..., b39, a0
            cyc = []
            for i in range(a_count):
                cyc.append(i)          # A vertex i
                cyc.append(20 + i)     # B vertex matched to i
            # Append remaining B vertices: 20+a_count .. 39
            for j in range(20 + a_count, 40):
                cyc.append(j)
            cycles.append(cyc)
    
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
