"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Square 1
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2
    # Known orthogonal pair for order 10
    s2 = [
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [2, 4, 6, 8, 1, 3, 5, 7, 9, 0],
        [4, 6, 8, 1, 3, 5, 7, 9, 0, 2],
        [6, 8, 1, 3, 5, 7, 9, 0, 2, 4],
        [8, 1, 3, 5, 7, 9, 0, 2, 4, 6],
        [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        [3, 5, 7, 9, 0, 2, 4, 6, 8, 1],
        [5, 7, 9, 0, 2, 4, 6, 8, 1, 3],
        [7, 9, 0, 2, 4, 6, 8, 1, 3, 5],
        [9, 0, 2, 4, 6, 8, 1, 3, 5, 7]
    ]
    
    return [s1, s2]
# EVOLVE-BLOCK-END
