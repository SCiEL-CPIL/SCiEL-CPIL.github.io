"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse pancyclic graph on N=40 vertices.
    Uses a Hamiltonian cycle (40 edges) and 19 chords from vertex 0 to vertices 2..20.
    Total edges: 59.
    Score: (59 - 40) / (77 - 40) ? No, score is seed_value / current_value.
    Seed value: 37.
    Current value: 19.
    Score: 37 / 19 approx 1.947.
    """
    edges = []
    
    # 1. Hamiltonian Cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
        
    # 2. Chords to ensure pancyclicity
    # Chord (0, k) creates cycles of length k+1 and N-k+1.
    # We need lengths 3..39.
    # k=2 gives 3 and 39.
    # k=3 gives 4 and 38.
    # ...
    # k=20 gives 21 and 21.
    # k=1 is edge (0,1) already in cycle.
    # k=39 is edge (0,39) already in cycle.
    for k in range(2, N // 2 + 1):
        edges.append((0, k))
        
    # Ensure distinctness (though logic guarantees it)
    # The set of edges is { (i, i+1) } U { (0, k) for 2<=k<=20 }
    # (0,1) and (0,39) are in the cycle.
    # (0,k) for 2<=k<=20 are distinct from cycle edges.
    # Cycle edges are (i, i+1). (0,2) is not (0,1) or (39,0).
    # So all edges are distinct.
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            d = L - 1
            if d <= N // 2:
                # Cycle 0-1-...-d-0
                # Vertices: 0, 1, ..., d
                cycle = list(range(d + 1))
            else:
                # Cycle 0-(N-1)-...-d'-0 where d' = N+1-L
                d_prime = N + 1 - L
                # Vertices: 0, N-1, N-2, ..., d_prime
                # range(N-1, d_prime-1, -1) generates N-1 down to d_prime
                cycle = [0] + list(range(N - 1, d_prime - 1, -1))
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
