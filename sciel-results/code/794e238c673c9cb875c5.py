"""
Sparse pancyclic graph on 40 vertices.
Strategy:
1. Base Hamiltonian cycle 0-1-...-39-0 (40 edges).
2. Add chords from hub vertex 0 to vertices 2..20 (19 edges).
   This creates cycles of length d+1 and N-d+1 for each chord (0, d).
   For d in 2..20:
     - Short cycle: 0-1-...-d-0 (length d+1) -> covers lengths 3..21
     - Long cycle:  0-39-38-...-d-0 (length N-d+1) -> covers lengths 21..39
   Combined with the Hamiltonian cycle (length 40), all lengths 3..40 are covered.
Total edges: 40 + 19 = 59. Value: 59 - 40 = 19.
"""

N = 40

def construct():
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Add chords from 0 to vertices 2..20
    chords = []
    for d in range(2, 21):
        edges.append((0, d))
        chords.append(d)
    
    # Build cycles for lengths 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        else:
            # Find which chord gives this cycle length
            # Short side: length = d + 1 => d = L - 1
            # Long side: length = N - d + 1 => d = N - L + 1
            d_short = L - 1
            d_long = N - L + 1
            
            if 2 <= d_short <= 20:
                # Use short side: 0-1-2-...-d-0
                cyc = list(range(d + 1))  # [0, 1, ..., d]
                # This is already a cycle since (0,d) is an edge and (0,1) is an edge
            elif 2 <= d_long <= 20:
                # Use long side: 0-39-38-...-d-0
                # Vertices: 0, N-1, N-2, ..., d
                cyc = [0] + list(range(N - 1, d - 1, -1))
            else:
                # Should not happen with our chord set
                raise ValueError(f"No chord for cycle length {L}")
        
        cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
