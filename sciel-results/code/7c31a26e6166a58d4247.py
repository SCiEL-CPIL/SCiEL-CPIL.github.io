N = 40

def construct():
    # Base cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i+1) % N))
    
    # Add chords (0, j) for j = 2 to 20
    for j in range(2, 21):
        edges.add((0, j))
    
    # Convert to list of tuples, normalize for undirected
    edge_list = []
    for e in edges:
        a, b = e
        if a > b:
            a, b = b, a
        edge_list.append((a, b))
    
    # Generate cycles of length 3 to 40
    cycles = []
    for L in range(3, N+1):
        if L == N:
            # Full cycle
            cycles.append(list(range(N)))
        elif L <= 21:
            # Use cycle 1 of chord (0, L-1): 0, 1, 2, ..., L-1
            cycles.append(list(range(L)))
        else:
            # Use cycle 2 of chord (0, j) where j = N+1-L
            j = N + 1 - L
            # Cycle: 0, j, j+1, ..., N-1
            cycle = [0] + list(range(j, N))
            cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
