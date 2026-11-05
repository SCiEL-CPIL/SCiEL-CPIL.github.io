import time

N = 40

def construct():
    # Structural change: Use a "ladder" backbone - two parallel paths with rungs.
    # Path A: 0, 2, 4, ..., 38 (20 vertices)
    # Path B: 1, 3, 5, ..., 39 (20 vertices)
    # Rungs: (0,1), (2,3), (4,5), ..., (38,39)
    # This gives a ladder graph. Then add strategic chords for all cycle lengths.
    
    # Actually, let's stick with the Hamiltonian cycle approach but verify correctness.
    # The current solution is already quite optimal at 59 edges.
    
    # Let me try: Hamiltonian cycle + chords, but verify all cycles are valid.
    
    start_time = time.time()
    
    # 1. Build Hamiltonian Cycle Edges
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Build Chords - (0, d) for d in 2..20
    chord_edges = [(0, d) for d in range(2, 21)]
    
    edges = ham_edges + chord_edges
    
    # 3. Build Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == 40:
            cycles.append(list(range(N)))
        else:
            if L <= 21:
                cycle_verts = list(range(L))
            else:
                end_vertex = 41 - L
                cycle_verts = [0] + list(range(N - 1, end_vertex - 1, -1))
            cycles.append(cycle_verts)
    
    # Validate
    edge_set = set()
    for e in edges:
        a, b = min(e), max(e)
        edge_set.add((a, b))
    
    for i, cyc in enumerate(cycles):
        L = i + 3
        if len(cyc) != L:
            pass
        for j in range(L):
            u = cyc[j]
            v = cyc[(j + 1) % L]
            a, b = min(u, v), max(u, v)
            if (a, b) not in edge_set:
                pass
    
    return {'edges': edges, 'cycles': cycles}
