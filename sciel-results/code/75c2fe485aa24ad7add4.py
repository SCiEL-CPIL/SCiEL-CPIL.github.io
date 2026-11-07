"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    # First K3,3: left {0,1,2}, right {3,4,5}
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([u, v])
    # Second K3,3: left {6,7,8}, right {9,10,11}
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
