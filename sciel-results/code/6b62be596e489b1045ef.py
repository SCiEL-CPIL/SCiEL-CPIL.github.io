N = 40

def construct():
    cycle_edges = [(i, (i+1) % N) for i in range(N)]
    chords = [(0, k) for k in range(2, 21)]
    edges = cycle_edges + chords
    
    cycles = []
    for length in range(3, N+1):
        if length == N:
            cycles.append(list(range(N)))
        elif length <= 21:
            # Forward arc of chord (0, length-1)
            cycles.append(list(range(length)))
        else:
            # Backward arc of chord (0, 41-length)
            k = 41 - length
            cycles.append([0] + list(range(39, k-1, -1)))
    
    return {'edges': edges, 'cycles': cycles}
