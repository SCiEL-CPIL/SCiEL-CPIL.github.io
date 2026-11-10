"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    es = set()
    # First K3,3 component: left {0,1,2}, right {3,4,5}
    for u in range(3):
        for v in range(3, 6):
            es.add(tuple(sorted((u, v))))
    # Second K3,3 component: left {6,7,8}, right {9,10,11}
    for u in range(6, 9):
        for v in range(9, 12):
            es.add(tuple(sorted((u, v))))
    return [list(e) for e in sorted(es)]
# EVOLVE-BLOCK-END
