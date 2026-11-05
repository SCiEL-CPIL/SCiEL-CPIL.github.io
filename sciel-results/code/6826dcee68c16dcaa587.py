import time

N = 40

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    edges = []
    edge_set = set()
    
    # 1. Add Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
            
    # 2. Add Chords from 0 to k for k in 2..20
    # This creates cycles of length k+1 and 41-k
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21, 21
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
            
    # 3. Generate Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Determine which chord to use
            # If L <= 21, use chord (0, L-1)
            # Path: 0 -> 1 -> ... -> L-1 -> 0
            # Length: (L-1) + 1 = L
            if L <= 21:
                k = L - 1
                # Cycle vertices: 0, 1, ..., k
                cycle = list(range(k + 1))
            else:
                # If L > 21, use chord (0, 41-L)
                # Let k = 41 - L
                # Path: 0 -> 39 -> 38 -> ... -> k -> 0
                # Length: (39 - k + 1) + 1 = 41 - k = L
                k = 41 - L
                # Cycle vertices: 0, 39, 38, ..., k
                cycle = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
