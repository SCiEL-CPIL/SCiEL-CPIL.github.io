import time

def construct():
    N = 40
    edges = set()
    
    # Base cycle C_N
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Chords from 0
    # We need chords for k in 2..19 and k=20
    # This covers all required cycle lengths
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list of tuples
    edge_list = list(edges)
    
    cycles = []
    # Lengths 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # The full cycle
            cycles.append(list(range(N)))
        elif L == 21:
            # k=20 case
            # Cycle 0-1-...-20-0
            cycles.append(list(range(21)))
        elif L <= 20:
            # k = L - 1
            # Cycle 0-1-...-k-0
            k = L - 1
            cycles.append(list(range(k + 1)))
        else:
            # L > 21
            # k = 41 - L
            # Cycle 0-k-(k+1)-...-39-0
            k = 41 - L
            # Vertices: 0, k, k+1, ..., 39
            cyc = [0] + list(range(k, N))
            cycles.append(cyc)
            
    return {'edges': edge_list, 'cycles': cycles}
