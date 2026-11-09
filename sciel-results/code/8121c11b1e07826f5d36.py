"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Hardcoded solution with 8 edges
    added = [
        [0, 4],
        [4, 8],
        [4, 12],
        [8, 16],
        [12, 20],
        [16, 20],
        [20, 23],
        [0, 23]
    ]
    return added
# EVOLVE-BLOCK-END
