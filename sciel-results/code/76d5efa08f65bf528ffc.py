"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices using a circular difference set model.
    
    Strategy:
    1. Start with a Hamiltonian cycle (40 edges) on vertices 0..39.
    2. Add chords to cover all cycle lengths 3..39.
    3. A chord with span d (distance along the Hamiltonian cycle) creates cycles of length
       d+1 (going the short way) and N-d+1 (going the long way).
    4. We need to cover all cycle lengths 3..39, which corresponds to spans 2..38.
    5. Using a symmetric chord placement (chord from i to i+d for i=0, covering spans d and N-d),
       we can cover all required spans with 19 chords (spans 2..20).
    6. Total edges: 40 (Hamiltonian) + 19 (chords) = 59.
    
    For each cycle length L in 3..40:
    - If L == 40: use the Hamiltonian cycle.
    - If L < 40: use a chord with span d = L-1 (or N-(L-1) if smaller), and construct the
      cycle as a path along the Hamiltonian cycle plus the closing chord.
    """
    n = N
    
    # Hamiltonian cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = []
    for i in range(n):
        j = (i + 1) % n
        if i < j:
            edges.append((i, j))
        else:
            edges.append((j, i))
    
    # Add chords: for each span d in 2..20, add chord (0, d)
    # This chord has span d (short way) and n-d (long way)
    # It covers cycle lengths d+1 and n-d+1
    chords = []
    for d in range(2, n // 2 + 1):  # d = 2 to 20
        chord = (0, d)
        chords.append((d, chord))  # store (span, chord)
        edges.append(chord)
    
    # Remove duplicates and ensure canonical form
    edge_set = set()
    for e in edges:
        if e[0] > e[1]:
            e = (e[1], e[0])
        edge_set.add(e)
    edges = sorted(edge_set)
    
    # Build adjacency for cycle construction
    adj = {}
    for u, v in edge_set:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    # Map span to chord
    span_to_chord = {}
    for d, chord in chords:
        span_to_chord[d] = chord
        span_to_chord[n - d] = chord
    
    # Construct cycles for each length L = 3..n
    cycles = []
    
    for L in range(3, n + 1):
        if L == n:
            # Hamiltonian cycle
            cycles.append(list(range(n)))
        else:
            # We need a cycle of length L
            # Use a chord with span d = L-1
            d = L - 1
            if d > n - d:
                d = n - d  # use the shorter span
            
            # Get the chord for this span
            chord = span_to_chord[d]
            u, v = chord
            
            # Construct the cycle:
            # If d = L-1, we go from u to v along the Hamiltonian cycle in d steps
            # The path is u, u+1, ..., u+d (mod n) = v
            # Cycle: u, u+1, ..., v (which is u+d mod n)
            
            # Determine the direction
            # Chord is (0, d) where d is in 2..20
            # If we use span d, the path goes 0, 1, 2, ..., d
            # If we use span n-d, the path goes 0, n-1, n-2, ..., n-d
            
            if d == L - 1:
                # Use the short way: 0, 1, 2, ..., d
                cycle = list(range(d + 1))
            else:
                # d = n - (L - 1), so L - 1 = n - d
                # Use the long way: 0, n-1, n-2, ..., n-d
                # This has length n - d + 1 = L
                cycle = [0] + list(range(n - 1, n - d - 1, -1))
            
            cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}

# EVOLVE-BLOCK-END
