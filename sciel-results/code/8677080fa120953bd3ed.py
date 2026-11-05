"""Construct a sparse pancyclic graph on 40 vertices."""
N = 40

def construct():
    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add chords at distances 2, 3, 4, ..., 20 from vertex 0
    # Chord (0, d) for d in 2..20
    for d in range(2, 21):
        edges.add((0, d))
    
    # Build cycles for lengths 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Full Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Use chord (0, k-1) to close a k-cycle: 0, 1, 2, ..., k-1, 0
            cycle = list(range(k))
        cycles.append(cycle)
    
    return {'edges': [tuple(sorted(e)) for e in edges], 'cycles': cycles}
