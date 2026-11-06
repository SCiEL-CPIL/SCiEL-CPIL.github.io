import time

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Uses a base cycle of 40 vertices and 19 chords from vertex 0 to vertices 2..20.
    This creates cycles of all lengths 3..40.
    Total edges: 40 (base) + 19 (chords) = 59.
    Score: 59 - 40 = 19.
    """
    N = 40
    edges = set()
    
    # 1. Base Cycle: 0-1-2-...-39-0
    for i in range(N):
        u, v = i, (i + 1) % N
        edges.add((min(u, v), max(u, v)))
        
    # 2. Chords: 0-k for k in 2..20
    # These chords create cycles of length k+1 and 41-k
    # k=2 -> lengths 3, 39
    # k=3 -> lengths 4, 38
    # ...
    # k=19 -> lengths 20, 22
    # k=20 -> length 21
    for k in range(2, 21):
        u, v = 0, k
        edges.add((min(u, v), max(u, v)))
        
    # Convert set to list of lists
    edge_list = [list(e) for e in sorted(edges)]
    
    # 3. Construct Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Base cycle
            cycle = list(range(N))
        elif L <= 20:
            # Cycle using chord (0, L-1) and path 0..L-1
            # Vertices: 0, 1, ..., L-1
            cycle = list(range(L))
        else:
            # Cycle using chord (0, d) and path d..39,0
            # Length L = 41 - d => d = 41 - L
            d = 41 - L
            # Vertices: d, d+1, ..., 39, 0
            cycle = list(range(d, N)) + [0]
        cycles.append(cycle)
        
    return {'edges': edge_list, 'cycles': cycles}
