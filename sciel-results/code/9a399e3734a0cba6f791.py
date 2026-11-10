def _known_cage(k, g):
    if k == 3 and g == 5:
        # Petersen
        n = 10
        edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                 (5,7), (7,9), (9,6), (6,8), (8,5),
                 (0,5), (1,6), (2,7), (3,8), (4,9)]
        return n, edges
    if k == 3 and g == 6:
        # Heawood
        n = 14
        edges = []
        for i in range(7):
            for j in range(7):
                if (j - i) % 7 in [1, 2, 4]:
                    edges.append((i, 7+j))
        return n, edges
    # Other small cages would be added here if edges are known
    return None
