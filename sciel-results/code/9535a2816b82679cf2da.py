N = 40

def construct():
    # Backbone: 40-cycle
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords: (0, d) for d in 2..20
    chord_edges = [(0, d) for d in range(2, 21)]
    
    edges = cycle_edges + chord_edges
    
    # Build cycles of each length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycle = list(range(N))
        elif k <= 21:
            d = k - 1
            cycle = list(range(d + 1))
        else:
            d = 41 - k
            cycle = [0] + list(range(d, N))
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
