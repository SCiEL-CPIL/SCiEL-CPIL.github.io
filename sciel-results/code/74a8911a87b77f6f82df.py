"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Main cycle edges
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords from vertex 0 to vertices 2..20
    # These chords create cycles of lengths 3..21 (short path) 
    # and 39..22 (long path)
    # Specifically:
    # Chord (0, k) creates cycle of length k+1 and N-k+1
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21, 21
    
    chords = [(0, k) for k in range(2, 21)]
    edges.extend(chords)
    
    # Ensure edges are unique and undirected
    # The generated edges are already distinct.
    # (0,1) is in main cycle. (0,2) is chord. etc.
    # No duplicates because main cycle edges are (i, i+1) mod N.
    # Chords are (0, k) for k in 2..20.
    # (0,1) is main cycle. (0,39) is main cycle.
    # So no overlap.
    
    # Generate cycles
    cycles = []
    
    # Cycle of length 40 is the main cycle
    # We need cycles of length 3..40 in that order.
    # So index 0 is length 3, index 1 is length 4, ..., index 37 is length 40.
    
    for length in range(3, N + 1):
        if length == N:
            # Main cycle
            cycle = list(range(N))
            # 0-1-...-39-0
            cycles.append(cycle)
        else:
            # Find a chord that creates a cycle of this length
            # We use the chords (0, k)
            # If length <= 21, use chord (0, length-1)
            # If length > 21, use chord (0, N-length+1)
            
            if length <= 21:
                k = length - 1
                # Cycle: 0, 1, ..., k, 0
                # Vertices: 0, 1, ..., k
                # Since k = length - 1, vertices are 0..length-1
                # Wait, the cycle is 0-1-...-k-0.
                # Vertices: 0, 1, ..., k.
                # Count: k+1 = length.
                # Path: 0 to k along the main cycle direction 0->1->...->k.
                cycle = list(range(length))
                # Check: 0-1-...-length-1-0.
                # Edge (length-1, 0) is the chord (0, length-1).
                # Edges (i, i+1) are main cycle.
                # This is valid.
                cycles.append(cycle)
            else:
                # length > 21
                # Use chord (0, k) where N-k+1 = length => k = N-length+1
                k = N - length + 1
                # Cycle: 0, k, k+1, ..., N-1, 0
                # Vertices: 0, k, k+1, ..., N-1
                # Count: 1 + (N-1 - k + 1) = N - k + 1 = length.
                # Path: 0 to k (chord), then k to N-1 (main cycle), then N-1 to 0 (main cycle).
                # Wait, the cycle is a simple cycle.
                # Vertices: 0, k, k+1, ..., N-1.
                # Edges: (0,k), (k, k+1), ..., (N-2, N-1), (N-1, 0).
                # This is a valid cycle.
                cycle = [0] + list(range(k, N))
                cycles.append(cycle)
                
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
