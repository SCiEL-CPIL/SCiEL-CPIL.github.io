import itertools

def construct():
    n = 16
    half = n // 2
    
    # K_{8,8} is the unique extremal triangle-free graph on 16 vertices.
    # It has diameter 2 and every edge is critical.
    # Max edges = 64.
    
    # Generate edges using itertools.product for structural distinction
    u_range = range(half)
    v_range = range(half, n)
    
    edges = []
    for u, v in itertools.product(u_range, v_range):
        edges.append([u, v])
        
    return edges
