import time

def construct():
    """
    Returns a graph with n=32 vertices, no 5-cycle, and independence number 8.
    Construction: 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    alpha = 8 * 1 = 8.
    No C5 because K4 has no C5 and components are disjoint.
    """
    # 8 disjoint K4s
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
