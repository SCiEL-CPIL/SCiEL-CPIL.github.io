def construct():
    N = 40
    edges = []
    edge_set = set()
    
    # 1. Add Hamiltonian Cycle
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v: u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
            
    # 2. Add Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        u, v = 0, k
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
            
    # 3. Generate Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        else:
            if L <= 21:
                # k = L - 1
                # Cycle 0, 1, ..., L-1
                cycle = list(range(L))
            else:
                # k = 41 - L
                # Cycle 0, 39, ..., k
                k = 41 - L
                cycle = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
