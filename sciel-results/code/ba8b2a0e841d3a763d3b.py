import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    This is a proven valid configuration for R(C5, K9) > 32.
    
    Properties:
    - n = 32
    - No 5-cycles (each component is K4, which has no cycles of length > 3)
    - No independent set of 9 (max independent set is 8, one from each K4)
    """
    n = 32
    edges = []
    # 8 disjoint blocks of size 4
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
