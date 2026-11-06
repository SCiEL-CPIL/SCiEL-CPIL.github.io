import time

N = 40

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Uses a base cycle C_40 and 19 chords from vertex 0 to vertices 2..20.
    This ensures cycles of all lengths 3..40.
    Total edges: 59.
    Score: 59 - 40 = 19.
    """
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s limit
    
    edges = set()
    
    # 1. Base Cycle C_40
    for i in range(N):
        edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
        
    # 2. Chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((min(0, k), max(0, k)))
        
    cycles = []
    
    # 3. Construct cycles for lengths 3..40
    for L in range(3, N + 1):
        if L == N:
            # Full cycle 0-1-...-39-0
            cyc = list(range(N))
        elif L <= 21:
            # Cycle 0-1-...-(L-1)-0 using chord (0, L-1)
            # Vertices are 0, 1, ..., L-1
            # Edges: (0,1), (1,2), ..., (L-2, L-1), (L-1, 0)
            # (L-1, 0) is a chord if L-1 > 1, i.e., L > 2.
            # For L=3, chord is (0,2).
            # For L=21, chord is (0,20).
            cyc = list(range(L))
        else:
            # Cycle 0-k-(k+1)-...-39-0 using chord (0, k)
            # Length = 1 (vertex 0) + (N - k) vertices from k to N-1
            # L = 1 + N - k  => k = N + 1 - L
            # Since N=40, k = 41 - L
            # Example L=22: k=19. Cycle 0-19-20-...-39-0. Length 1+21=22.
            # Chord is (0, 19).
            # Example L=39: k=2. Cycle 0-2-3-...-39-0. Length 1+38=39.
            # Chord is (0, 2).
            k = N + 1 - L
            cyc = [0] + list(range(k, N))
            
        cycles.append(cyc)
    
    # Validation check (fast)
    # We know this construction is valid, but let's be safe.
    # The cycles are simple and distinct vertices.
    # The edges required are:
    # 1. Cycle edges: (i, i+1) mod 40. All present in base cycle.
    # 2. Chords: (0, L-1) for L in 3..21. So (0, 2)..(0, 20).
    # 3. Chords: (0, 41-L) for L in 22..39. So (0, 19)..(0, 2).
    #    Note: (0, 20) is used for L=21. (0, 19) is used for L=22.
    #    The set of chords needed is { (0,2), ..., (0,20) }.
    #    We added exactly these.
    
    return {
        'edges': list(edges),
        'cycles': cycles
    }
