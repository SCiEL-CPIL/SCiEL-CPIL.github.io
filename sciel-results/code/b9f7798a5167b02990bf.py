import time

def construct():
    """
    Returns the edges of the Clebsch graph on 16 vertices.
    The Clebsch graph is a triangle-free graph with chromatic number 4
    and 40 edges. It is the unique strongly regular graph srg(16, 5, 0, 2).
    """
    n = 16
    edges = []
    
    # Connection set for the Clebsch graph in Z_2^4
    # Vertices are 4-bit integers 0..15
    # Two vertices u, v are adjacent if u XOR v is in S
    S = [1, 2, 4, 8, 15]  # 0001, 0010, 0100, 1000, 1111
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u ^ v) in S:
                edges.append([u, v])
                
    return edges
