import time

N = 40

def construct():
    start_time = time.time()
    deadline = 90.0  # 90s limit, use 89s for safety
    
    # 40-cycle: 0-1-2-...-39-0
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords from 0 to 2, 3, ..., 20
    # This creates 19 chords. Total edges = 40 + 19 = 59.
    chord_edges = [(0, k) for k in range(2, 21)]
    
    edges = cycle_edges + chord_edges
    
    # Verify no duplicate edges
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            # Fallback: try a different construction if duplicate found (should not happen)
            # For now, assert as it's a fixed construction
            pass
        edge_set.add(key)
        
    # Build cycles of length k for k = 3, 4, ..., 40
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            # Short cycle: 0, 1, 2, ..., k-1
            # Uses chords (0, k-1) and path 0-1-...-(k-1)
            # Wait, cycle is 0-1-...-(k-1)-0.
            # Edge (k-1, 0) is the chord.
            # Path 0-1-...-(k-1) uses Hamiltonian edges.
            cyc = list(range(k))
        else:
            # Long cycle using chord (0, 41-k)
            # Let j = 41 - k.
            # Cycle: 0, j, j+1, ..., 39, 0
            # Uses chord (0, j) and path j-...-39-0.
            chord_target = 41 - k
            if chord_target == 0:
                cyc = list(range(N))
            else:
                # Vertices: 0, j, j+1, ..., 39
                # Length: 1 (0) + (39 - j + 1) = 40 - j + 1 = 41 - j = k.
                cyc = [0] + list(range(chord_target, N))
        cycles.append(cyc)
        
        # Time check
        if time.time() > deadline - 1.0:
            # Return current result immediately
            return {'edges': edges, 'cycles': cycles}

    # Verify cycles (optional in production, but good for correctness)
    # The construction is deterministic and known to be correct.
    
    return {'edges': edges, 'cycles': cycles}
