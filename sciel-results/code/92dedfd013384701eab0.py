import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    # 1. Construct Edges
    edges = []
    
    # Base cycle (rim)
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from hub 0 to vertices 2..20
    # These chords create cycles of lengths 3..20 (short arc) and 39..21 (long arc)
    for j in range(2, 21):
        edges.append((0, j))
        
    # Total edges: 40 (rim) + 19 (chords) = 59
    # Objective: 59 - 40 = 19
    
    # 2. Construct Cycles
    cycles = []
    
    for k in range(3, N + 1):
        if k <= 20:
            # Cycle: 0-1-...-(k-1)-0
            # Vertices: 0, 1, ..., k-1
            cyc = list(range(k))
        elif k < N:
            # Cycle: 0-j-(j+1)-...-39-0
            # Length k corresponds to chord (0, j) where j = 41 - k
            j = 41 - k
            # Vertices: 0, j, j+1, ..., 39
            # Note: range(j, N) gives j..39
            cyc = [0] + list(range(j, N))
        else:
            # k == 40
            # Cycle: 0-1-...-39-0
            cyc = list(range(N))
            
        cycles.append(cyc)
        
    # Check time
    if time.time() > deadline:
        # If we somehow timed out (shouldn't happen with static construction), return best effort
        pass

    return {'edges': edges, 'cycles': cycles}
