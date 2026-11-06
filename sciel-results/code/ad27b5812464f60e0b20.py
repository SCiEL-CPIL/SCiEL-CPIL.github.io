import time

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Returns a dictionary with 'n' and 'edges'.
    
    The graph is the disjoint union of 8 copies of K4.
    n = 32.
    alpha(G) = 8.
    C5-free.
    """
    n = 32
    edges = []
    
    # 8 cliques of size 4
    for b in range(8):
        start = 4 * b
        # Generate all edges within the clique {start, start+1, start+2, start+3}
        for i in range(4):
            for j in range(i + 1, 4):
                u = start + i
                v = start + j
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
