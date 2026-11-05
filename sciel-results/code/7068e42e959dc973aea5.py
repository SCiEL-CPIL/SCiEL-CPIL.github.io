"""Fresh design: Hamiltonian cycle + 18 chords from vertex 0 to cover all cycle lengths 3-39."""
N = 40

def construct():
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 18 chords from vertex 0 to vertices 2..19
    # Chord (0,k) creates cycles of length k+1 and 41-k
    # k=2 -> lengths 3, 39
    # k=3 -> lengths 4, 38
    # ...
    # k=19 -> lengths 20, 21
    for k in range(2, 20):
        edges.append((0, k))
    
    # Build cycles for lengths 3..40
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            # Use chord (0, k) where k+1 = length or 41-k = length
            # If length <= 20: use k = length - 1, cycle is 0,1,...,k
            # If length > 20: use k = 41 - length, cycle is 0,39,38,...,k
            if length <= 20:
                k = length - 1
                # Cycle: 0 -> 1 -> 2 -> ... -> k -> 0
                cyc = list(range(k + 1))
            else:
                k = 41 - length
                # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0
                cyc = [0] + list(range(39, k - 1, -1))
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
