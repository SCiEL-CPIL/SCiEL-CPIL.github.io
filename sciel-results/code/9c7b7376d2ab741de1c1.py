import time

N = 40

def construct():
    # Base cycle edges
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
    
    # Chords
    # (0, k) for k in 2..19
    for k in range(2, 20):
        edges.add((0, k))
    # (1, 39)
    edges.add((1, 39))
    
    edge_list = list(edges)
    
    # Construct cycles
    cycles = {}
    
    # Length 40: Base cycle
    cycles[40] = list(range(N))
    
    # Lengths 3..20: Forward path of chord (0, L-1)
    for L in range(3, 21):
        k = L - 1
        # Cycle: 0, 1, ..., k
        cycles[L] = list(range(k+1))
        
    # Lengths 21..37: Backward path of chord (0, 40-L)
    for L in range(21, 38):
        k = 40 - L
        # Cycle: 0, 39, 38, ..., k
        # Vertices: 0, N-1, N-2, ..., k
        path = [0] + list(range(N-1, k-1, -1))
        cycles[L] = path
        
    # Length 38: Backward path of chord (0, 3)
    # k=3, L=38. Path 0, 39, ..., 3
    cycles[38] = [0] + list(range(N-1, 2, -1))
    
    # Length 39: Forward path of chord (1, 39)
    # Cycle: 1, 2, ..., 39
    cycles[39] = list(range(1, N))
    
    # Format output
    return {
        'edges': edge_list,
        'cycles': [cycles[L] for L in range(3, N+1)]
    }

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Time: {time.time() - start}")
    print(f"Edges: {len(res['edges'])}")
    print(f"Score: {77 / len(res['edges'])}")
