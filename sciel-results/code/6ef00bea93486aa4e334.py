"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Uses a Hamiltonian cycle and 19 chords from a hub vertex to cover all cycle lengths 3..39.
    Total edges: 40 (cycle) + 19 (chords) = 59.
    This is the theoretical minimum for a pancyclic graph on 40 vertices.
    """
    # Vertices are 0..39
    # Base Hamiltonian Cycle: 0-1-2-...-39-0
    edges = []
    edge_set = set()
    
    def add_edge(u, v):
        if u == v:
            raise ValueError("Self-loop")
        e = (min(u, v), max(u, v))
        if e in edge_set:
            raise ValueError(f"Duplicate edge {e}")
        edge_set.add(e)
        edges.append((u, v))

    # Add Hamiltonian cycle edges
    for i in range(N):
        add_edge(i, (i + 1) % N)
    
    # Add chords from vertex 0 to k, for k in 2..20
    # Chord (0, k) creates cycles of length k+1 and N-k+1
    # We need to cover lengths 3..39
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21, 21
    for k in range(2, 21):
        add_edge(0, k)
        
    # Construct cycles for lengths 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Find k such that k+1 == L or N-k+1 == L
            # k+1 == L => k = L-1
            # N-k+1 == L => k = N-L+1
            # Check if k is in our chords list (2..20)
            k1 = L - 1
            k2 = N - L + 1
            
            found = False
            if 2 <= k1 <= 20:
                # Cycle using chord (0, k1) and path 0->1->...->k1->0
                # Length is k1 + 1 = L
                cycle = list(range(0, k1 + 1))
                found = True
            elif 2 <= k2 <= 20:
                # Cycle using chord (0, k2) and path 0->39->...->k2->0
                # Length is N - k2 + 1 = L
                # Path is 0, N-1, ..., k2, 0
                # Vertices: 0, N-1, N-2, ..., k2
                # Order: 0, N-1, N-2, ..., k2
                cycle = [0] + list(range(N - 1, k2 - 1, -1))
                found = True
            
            if not found:
                raise ValueError(f"Could not construct cycle of length {L}")
        
        # Verify cycle validity
        # 1. Distinct vertices
        if len(cycle) != len(set(cycle)):
            raise ValueError(f"Cycle {L} has duplicate vertices")
        # 2. Length matches
        if len(cycle) != L:
            raise ValueError(f"Cycle {L} has wrong length {len(cycle)}")
        # 3. Edges exist
        for i in range(L):
            u = cycle[i]
            v = cycle[(i + 1) % L]
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                raise ValueError(f"Edge {e} missing for cycle {L}")
                
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
