import time

def construct():
    # Deadline guard
    start = time.time()
    deadline = start + 85
    
    n = 16
    half = n // 2
    
    # K_{8,8} is the unique extremal triangle-free graph on 16 vertices (Mantel's Theorem).
    # It has diameter 2 and every edge is critical (deleting one increases diameter to 3).
    # Max edges = 64.
    
    edges = [[u, v] for u in range(half) for v in range(half, n)]
    
    # Time check
    if time.time() > deadline:
        return edges
        
    return edges
