import time

N = 40

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Base cycle edges: (0,1), (1,2), ..., (39,0)
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Chords from 0 to 2..20: each chord (0,d) creates cycles of length d+1 and 41-d
    # d=2..19 covers lengths 3..20 and 39..22 (36 lengths)
    # d=20 covers length 21
    # Total: 37 cycle lengths (3..39) covered by 19 chords
    for d in range(2, 21):
        edges.add((0, d))
        
    edge_list = list(edges)
    
    # Generate cycles of lengths 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        else:
            d1 = L - 1      # forward arc length
            d2 = 41 - L     # backward arc length
            
            if 2 <= d1 <= 20:
                # Cycle: 0 -> 1 -> 2 -> ... -> d1 -> 0
                cycle = list(range(d1 + 1))
            else:
                # Cycle: 0 -> 39 -> 38 -> ... -> d2 -> 0
                cycle = [0] + list(range(39, d2 - 1, -1))
                
        cycles.append(cycle)
        
    return {'edges': edge_list, 'cycles': cycles}
