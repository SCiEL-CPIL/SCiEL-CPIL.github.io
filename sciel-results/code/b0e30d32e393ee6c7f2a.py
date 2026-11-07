import time

def construct():
    """
    Constructs a dense graph on 15 vertices avoiding equal-degree path endpoints.
    Uses a deterministic K_{7,8} construction which is provably valid.
    """
    n = 15
    # Partition vertices into two sets A and B
    # A = 0..6 (7 vertices), B = 7..14 (8 vertices)
    # Edges only between A and B
    
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    return edges
