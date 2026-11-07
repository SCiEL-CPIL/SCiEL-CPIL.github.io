"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit

    # Split Fano plane on vertices 0-8: 7 edges, 3-chromatic
    # Original Fano on 0-6: 012, 034, 056, 135, 146, 236, 245
    # Split vertex 0 into {0,7}, vertex 1 into {1,8}
    edges = [
        [7, 1, 2],   # was 012, split 0->7
        [0, 3, 4],   # was 034, kept 0
        [0, 5, 6],   # was 056, kept 0
        [8, 3, 5],   # was 135, split 1->8
        [8, 4, 6],   # was 146, split 1->8
        [2, 3, 6],   # was 236
        [2, 4, 5]    # was 245
    ]

    # Valid 3-coloring: no edge is monochromatic
    # Fano coloring: 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Split 0 (color 0) into 0 (color 0) and 7 (color 0)
    # Split 1 (color 1) into 1 (color 1) and 8 (color 1)
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 1]

    # Verify: every edge must be non-monochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # Fallback to seed if somehow invalid (should not happen)
            from itertools import combinations
            return {'edges': [list(e) for e in combinations(range(5), 3)],
                    'colors': [0, 0, 1, 1, 2, 0, 0, 0, 0]}

    # Verify types
    edges = [list(e) for e in edges]
    colors = list(colors)

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
