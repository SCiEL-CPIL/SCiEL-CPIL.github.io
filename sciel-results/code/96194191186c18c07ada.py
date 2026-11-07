import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Optimal solution derived from analysis:
    # Center at 11.
    # Add edges to cover all vertices within distance 2 of 11.
    # Left side (0-8): Edges to 0, 2, 4, 6, 8
    # Right side (14-23): Edges to 14, 16, 18, 20, 22
    
    edges_to_add = [
        [11, 0],
        [11, 2],
        [11, 4],
        [11, 6],
        [11, 8],
        [11, 14],
        [11, 16],
        [11, 18],
        [11, 20],
        [11, 22]
    ]
    
    # Convert to required format: list of [u, v] with Python ints
    result = []
    for u, v in edges_to_add:
        result.append([int(u), int(v)])
        
    return result
# EVOLVE-BLOCK-END
