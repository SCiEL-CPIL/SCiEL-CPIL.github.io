"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Build a Hamiltonian cycle: 0-1-2-...-39-0
    # Plus chords from vertex 0 to vertices 2,3,...,38 to create all cycle lengths 3-39
    
    edges = []
    edge_set = set()
    
    # Hamiltonian cycle edges
    for i in range(N):
        a, b = i, (i + 1) % N
        if a > b:
            a, b = b, a
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edges.append((a, b))
    
    # Chords from 0 to k for k=2,3,...,38
    # Chord (0,k) creates cycles of length k+1 (forward) and 40-k (backward)
    for k in range(2, N):
        a, b = 0, k
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edges.append((a, b))
    
    # Build cycles for lengths 3 through 40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Find k such that k+1 == L or 40-k == L
            # Prefer k = L-1 if L-1 < 40, else k = 40-L
            k = L - 1
            if k >= N:
                k = N - L
            # Forward cycle: 0, 1, 2, ..., k
            # This works when k+1 == L
            if k + 1 == L and k < N:
                cycle = list(range(L))  # 0, 1, 2, ..., L-1
            else:
                # Backward cycle: 0, 39, 38, ..., k
                # Length is 40 - k
                k = N - L
                cycle = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
