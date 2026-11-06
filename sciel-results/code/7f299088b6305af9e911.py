import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin
    
    # Current best known: 8 disjoint K4s
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Return the known valid construction
    return {'n': n, 'edges': edges}
