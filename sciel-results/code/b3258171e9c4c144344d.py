import time

N = 40

def construct():
    # Start time for safety, though this construction is O(1)
    start_time = time.time()
    
    # Base cycle edges
    base_edges = [(i, (i+1) % N) for i in range(N)]
    
    # Fan chords from 0 to 2..20
    chords = [(0, j) for j in range(2, 21)]
    
    edges = base_edges + chords
    
    # Ensure unique edges (they are by construction)
    # edges = list(set(tuple(sorted(e)) for e in edges)) # Not needed, but safe
    
    cycles = []
    
    # Cycles of length 3 to 21: 0, 1, ..., L-1, 0
    # Uses chord (0, L-1)
    for L in range(3, 22):
        cycles.append(list(range(L)))
        
    # Cycles of length 22 to 39: 0, j, j+1, ..., 39, 0
    # Length is 41 - j
    # We need L = 41 - j => j = 41 - L
    # For L in 22..39, j is 19..2
    for L in range(22, 40):
        j = 41 - L
        # Cycle: 0, j, j+1, ..., 39, 0
        cycle = [0]
        # Add j, j+1, ..., 39
        for v in range(j, N):
            cycle.append(v)
        cycles.append(cycle)
        
    # Cycle of length 40: 0, 1, ..., 39, 0
    cycles.append(list(range(N)))
    
    return {'edges': edges, 'cycles': cycles}
