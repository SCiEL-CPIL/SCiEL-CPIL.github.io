import time

N = 40

def construct():
    # Deadline guard (not strictly needed for deterministic O(1) construction, but good practice)
    start_time = time.time()
    
    edges = []
    
    # Base cycle: 0-1-2-...-39-0
    # Edges: (0,1), (1,2), ..., (39,0)
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to d, for d in 2..20
    # These create cycles of length d+1 (path 0..d + chord)
    # and length N-d+1 (path d..N-1..0 + chord)
    for d in range(2, 21):
        # Check if edge (0, d) already exists
        # Base edges are (i, i+1). (0, d) is not in base edges for d > 1 and d < N-1.
        # (0, 1) is base. (0, 39) is base.
        # d is 2..20, so no conflict.
        edges.append((0, d))
        
    # Total edges: 40 + 19 = 59
    # Score: 59 - 40 = 19
    
    cycles = []
    
    # Generate cycles for lengths 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # Cycle of length 40: 0-1-...-39-0
            cyc = list(range(N))
        else:
            d = L - 1
            if d <= 20:
                # Use chord (0, d)
                # Cycle: 0-1-...-d-0
                cyc = list(range(d + 1))
            else:
                # Use chord (0, 41-L)
                # Let d' = 41 - L
                # Cycle: 0-d'-...-39-0
                d_prime = 41 - L
                # Vertices: 0, d_prime, d_prime+1, ..., 39
                # Note: The cycle is 0 -> d_prime -> ... -> 39 -> 0
                # List: [0, d_prime, d_prime+1, ..., 39]
                cyc = [0] + list(range(d_prime, N))
        
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
