def construct():
    N = 40
    edges = []
    edges_set = set()
    
    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))
            
    # 1. Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        add_edge(i, (i + 1) % N)
        
    # 2. Chords from 0 to c for c in 2..20
    # These chords allow forming cycles of length c+1 and 41-c
    chords = list(range(2, 21)) # 2 to 20 inclusive
    for c in chords:
        add_edge(0, c)
        
    cycles = []
    
    # We need one cycle for each length L in 3..40
    for L in range(3, N + 1):
        if L == N:
            # Use the Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Check if L is covered by short arc of chord c = L-1
            c_short = L - 1
            if 2 <= c_short <= 20:
                # Cycle: 0, 1, ..., c_short
                cycle = [0] + list(range(1, c_short + 1))
            else:
                # Must be covered by long arc of chord c = 41-L
                c_long = 41 - L
                # c_long will be in 2..19 for L in 22..39
                # Cycle: 0, 39, 38, ..., c_long
                # Vertices: 0, then 39 down to c_long
                cycle = [0] + list(range(39, c_long - 1, -1))
                
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
