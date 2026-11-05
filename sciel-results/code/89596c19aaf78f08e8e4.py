import time

def construct():
    # Start time
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

    # Construct 8 disjoint copies of K4
    # Total vertices n = 32
    n = 32
    edges = []
    
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Check validity (optional but good for debugging, though we know it's valid)
    # n=32, alpha=8, no C5.
    
    return {'n': n, 'edges': edges}
