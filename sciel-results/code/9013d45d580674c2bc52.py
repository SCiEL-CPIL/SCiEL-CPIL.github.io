"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add(tuple(sorted((i, (i + 1) % N))))
    
    # Chords from vertex 0 to vertices 2..19 (18 chords) for cycles of length 3..20 and 38..21
    for j in range(2, 20):
        edges.add(tuple(sorted((0, j))))
    
    # Chord (1, 39) for the 39-cycle
    edges.add(tuple(sorted((1, 39))))
    
    edges = sorted(edges)
    
    # Build cycles for each length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == 40:
            # Full Hamiltonian cycle
            cyc = list(range(40))
        elif k == 39:
            # Cycle 1-2-...-39-1
            cyc = list(range(1, 40))
        elif k <= 20:
            # Cycle 0-1-2-...-(k-1)-0
            cyc = list(range(k))
        else:
            # 21 <= k <= 38: use chord (0, 40-k)
            # Cycle: 0, (40-k), (40-k+1), ..., 39, 0
            start = 40 - k
            cyc = [0] + list(range(start, 40))
        
        cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
