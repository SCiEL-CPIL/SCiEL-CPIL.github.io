import time

def construct():
    """
    Returns a graph with the maximum known number of vertices n (32)
    that is C5-free and has no independent set of size 9.
    
    The construction is the disjoint union of 8 copies of K4.
    n = 32
    alpha(G) = 8 < 9
    G is C5-free because K4 is C5-free and the union is disjoint.
    """
    # The known best construction for this problem is 8 disjoint K4s.
    # n = 32.
    # Score = 32 / 28 (seed) = 1.1428...
    
    num_copies = 8
    n = 4 * num_copies
    edges = []
    
    for b in range(num_copies):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
