N = 40

def construct():
    edges = set()
    
    # Base cycle: 0-1-2-...-39-0
    for i in range(N):
        edges.add((i, (i+1) % N))
    
    # Chords: (0, k) for k = 2, 3, ..., 19 (18 chords)
    for k in range(2, 20):
        edges.add((0, k))
    
    # Additional chord: (20, 39) - distance 19 on the base cycle
    edges.add((20, 39))
    
    # Convert to list of sorted tuples
    edge_list = sorted(tuple(sorted(e)) for e in edges)
    
    # Generate cycles
    cycles = []
    for L in range(3, N+1):
        if L == N:
            cyc = list(range(N))
        elif L == 21:
            # Use chord (0,2) and chord (20,39)
            # Cycle: 0 -> 2 (chord) -> 3 -> ... -> 20 (path) -> 39 (chord) -> 0 (path)
            cyc = [0, 2] + list(range(3, 21)) + [39]
        elif L <= 20:
            # Use chord (0, L-1) and path from L-1 back to 0
            # Cycle: 0 -> 1 -> ... -> L-1 (path) -> 0 (chord)
            # Wait, this gives length L if we go 0-1-...-(L-1)-0
            # Edges: (0,1), (1,2), ..., (L-2, L-1), (L-1, 0)
            # Length: L-1 + 1 = L
            cyc = list(range(L))
        else:  # 22 <= L <= 39
            # Use chord (0, L-1) and path from L-1 back to 0 going the long way
            # Path: L-1 -> L-2 -> ... -> 1 -> 0
            # Length: (L-1) + 1 = L
            # Vertices: 0, L-1, L-2, ..., 1
            # Wait, let's check: 0 -> L-1 (chord) -> L-2 -> ... -> 1 -> 0 (base edge)
            # Edges: (0, L-1), (L-1, L-2), ..., (2, 1), (1, 0)
            # Length: 1 + (L-2) + 1 = L
            # Vertices: 0, L-1, L-2, ..., 1
            cyc = [0] + list(range(L-1, 0, -1))
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
