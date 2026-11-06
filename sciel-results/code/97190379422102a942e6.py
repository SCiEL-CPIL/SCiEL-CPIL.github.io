import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Base cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords from 0 to k, for k = 2 to 20
    # These create cycles of length k+1 and N-k+1
    for k in range(2, N // 2 + 1):
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # Generate cycles
    cycles = []
    for L in range(3, N + 1):
        if time.time() > deadline:
            # Fallback: return what we have, though generation is fast
            break
            
        if L == N:
            # Full cycle
            cyc = list(range(N))
        elif L <= N // 2 + 1:
            # Use chord (0, L-1)
            # Cycle: 0, 1, ..., L-1, 0
            k = L - 1
            cyc = list(range(k + 1)) # 0 to L-1
            # The cycle is 0-1-...-L-1-0
            # Vertices: 0, 1, ..., L-1
        else:
            # Use chord (0, N-L+1)
            # Cycle: 0, N-L+1, N-L+2, ..., N-1, 0
            k = N - L + 1
            # Vertices: 0, k, k+1, ..., N-1
            cyc = [0] + list(range(k, N))
            
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}
