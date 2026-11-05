N = 40

def construct():
    edges = set()
    # Base 40-cycle
    for i in range(N):
        edges.add((i, (i+1) % N))
    
    # Chords from 0 to d for d=2..20
    for d in range(2, 21):
        edges.add((0, d))
    
    edge_list = sorted([tuple(sorted(e)) for e in edges])
    
    cycles = []
    for length in range(3, N+1):
        if length == N:
            cycles.append(list(range(N)))
        elif length <= 20:
            # d = length-1, cycle = [0, 1, ..., length-1]
            cycles.append(list(range(length)))
        elif length == 21:
            # d=20, cycle = [0, 1, 2, ..., 20]
            cycles.append(list(range(21)))
        else:
            # length = 22..39, d = 41-length
            d = 41 - length
            # Cycle: [0, 39, 38, ..., d]
            cycle = [0] + list(range(39, d-1, -1))
            cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
