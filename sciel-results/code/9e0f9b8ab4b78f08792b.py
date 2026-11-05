"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Hamiltonian cycle: 0-1-2-...-39-0
    hc_edges = [(i, (i+1) % N) for i in range(N)]
    
    # Add chords from 0 to close cycles of length 3..39
    # Cycle of length k: vertices 0,1,...,k-1, need edge (0, k-1)
    chord_edges = [(0, k-1) for k in range(3, N)]  # (0,2), (0,3), ..., (0,38)
    
    all_edges = hc_edges + chord_edges
    # Remove duplicates (keep as set for uniqueness)
    edge_set = set()
    for a, b in all_edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    edges = list(edge_set)
    
    # Build cycles for lengths 3..40
    cycles = []
    for k in range(3, N+1):
        if k == N:
            # Full Hamiltonian cycle
            cyc = list(range(N))
        else:
            # Vertices 0, 1, ..., k-1 with chord (0, k-1)
            cyc = list(range(k))
        cycles.append(cyc)
    
    # Verify: check all edges exist and cycles are valid
    edge_lookup = set()
    for a, b in edges:
        edge_lookup.add((a, b))
        edge_lookup.add((b, a))
    
    valid = True
    for cyc in cycles:
        k = len(cyc)
        for i in range(k):
            u = cyc[i]
            v = cyc[(i+1) % k]
            if (u, v) not in edge_lookup:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use seed construction
        return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
                'cycles': [list(range(k)) for k in range(3,N+1)]}
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
