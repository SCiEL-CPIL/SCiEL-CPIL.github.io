# EVOLVE-BLOCK-START

def construct():
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
