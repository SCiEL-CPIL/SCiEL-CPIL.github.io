import time

def construct():
    """
    Constructs a graph with no 5-cycle and no independent set of size 9.
    Uses 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    alpha(G) = 8 < 9.
    No C5 because components are K4.
    """
    # Deadline guard, though this construction is instantaneous
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds is plenty, we have 150s total
    
    k = 8 # Number of K4 components
    n = k * 4
    
    edges = []
    for b in range(k):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
