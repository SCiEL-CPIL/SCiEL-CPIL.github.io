"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        a, b = i, (i + 1) % N
        if a > b:
            a, b = b, a
        edges.add((a, b))
    
    # Chords (0, k) for k = 2..20
    for k in range(2, 21):
        edges.add((0, k))
    
    edges = sorted(edges)
    
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            cycles.append(list(range(N)))
        elif length <= 20:
            # Forward cycle: 0, 1, ..., length-1, 0
            cycles.append(list(range(length)))
        else:
            # Backward cycle: 0, N-1, N-2, ..., N-length, 0
            # Length = N - k + 1 where k = N - length + 1... 
            # We need cycle of given length. For length L > 20:
            # k = N - L + 1... no.
            # Chord (0,k) backward cycle has length N-k+1 = L => k = N-L+1
            k = N - length + 1
            # Cycle: 0, N-1, N-2, ..., k, 0
            cyc = [0] + list(range(N - 1, k - 1, -1))
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
