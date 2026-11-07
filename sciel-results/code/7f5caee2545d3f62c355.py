import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    edges = []
    
    # Generate all edges for K_16
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v])
            
    # Remove one edge to ensure diameter is exactly 2
    # Remove edge [0, 1]
    # The distance between 0 and 1 will be 2 (via any other node)
    # All other pairs have distance 1
    # Diameter is 2
    
    # Find and remove [0, 1]
    edges = [e for e in edges if not (e[0] == 0 and e[1] == 1)]
    
    return edges
