import time

N = 40

def construct():
    edges = set()
    
    # Base cycle
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Chords
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            if L <= 21:
                k = L - 1
                cycle_verts = list(range(k + 1))
            else:
                k = N - L + 1
                cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cycle_verts)
        
    return {'edges': [tuple(sorted(e)) for e in edges], 'cycles': cycles}
