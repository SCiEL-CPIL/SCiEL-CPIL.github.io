N = 40

def construct():
    # Base cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i+1) % N))
    
    # Chords: (0, k) for k = 2, 3, ..., 20
    for k in range(2, 21):
        edges.add((0, k))
    
    # Convert to list of sorted tuples
    edge_list = sorted(tuple(sorted(e)) for e in edges)
    
    # Generate cycles
    cycles = []
    for L in range(3, N+1):
        if L == N:
            cyc = list(range(N))
        elif L == 21:
            cyc = list(range(21))  # 0,1,...,20
        elif L <= 20:
            cyc = list(range(L))  # 0,1,...,L-1
        else:  # 22 <= L <= 39
            k = 41 - L
            cyc = [0] + list(range(k, N))
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
