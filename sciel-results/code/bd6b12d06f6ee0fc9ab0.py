"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # A 3 x 4 grid.
    # This configuration has 24 ordinary lines.
    # Points: (0,0) to (3,2)
    points = [[x, y] for x in range(4) for y in range(3)]
    return points
# EVOLVE-BLOCK-END
