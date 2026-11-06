N = 40

def construct():
    edges = []
    # Cycle edges
    for i in range(N):
        edges.append((i, (i+1) % N))
    # Chords from 0 to even vertices 2,4,...,38
    for i in range(2, N, 2):
        edges.append((0, i))
    
    cycles = []
    for L in range(3, N+1):
        if L == N:
            cycles.append(list(range(N)))
        elif L % 2 == 1:
            cycles.append(list(range(L)))
        else:
            cycles.append([0] + list(range(2, L+1)))
    
    return {'edges': edges, 'cycles': cycles}
