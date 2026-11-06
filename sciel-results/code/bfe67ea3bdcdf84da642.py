def construct():
    N = 40
    
    # 1. Base Cycle Edges: (i, i+1 mod N) for i in 0..39
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # 2. Chords from Hub 0: (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    edge_list = sorted(edges)
    
    # 3. Generate Cycles of length L for L in 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        elif L <= 21:
            # Cycle: 0, 1, 2, ..., L-1
            cycle_verts = list(range(L))
        else:
            # Cycle: 0, k, k+1, ..., 39 where k = 41-L
            k = N + 1 - L
            cycle_verts = [0] + list(range(k, N))
        cycles.append(cycle_verts)
    
    return {'edges': edge_list, 'cycles': cycles}
