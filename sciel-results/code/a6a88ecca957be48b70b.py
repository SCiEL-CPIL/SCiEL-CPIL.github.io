"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Valid 10-edge solution
    # Hub at 12
    # Leaves: 0, 2, 4, 6, 8 (evens left) and 15, 17, 19, 21, 23 (odds right)
    # This ensures max distance from hub is 2 for all nodes.
    # Diameter <= 4.
    # Triangle-free: No leaf pairs are adjacent in path.
    
    added_edges = [
        [12, 0],
        [12, 2],
        [12, 4],
        [12, 6],
        [12, 8],
        [12, 15],
        [12, 17],
        [12, 19],
        [12, 21],
        [12, 23]
    ]
    
    return added_edges
# EVOLVE-BLOCK-END
