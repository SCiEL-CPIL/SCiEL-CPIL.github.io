N = 40

def construct():
    edges = set()
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    # Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    edges_list = sorted(edges)
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle
            cycle = list(range(N))
        elif 3 <= L <= 21:
            # Chord (0, L-1): cycle 0, 1, 2, ..., L-1
            cycle = list(range(L))
        else:
            # L in 22..39: chord (0, k) where k = 41 - L
            k = 41 - L
            # Cycle: 0, 39, 38, ..., k
            cycle = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cycle)
    
    return {'edges': edges_list, 'cycles': cycles}
