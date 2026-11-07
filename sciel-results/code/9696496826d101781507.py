"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    # Use a stronger distance decay to encourage local moves.
    # This helps in filling the truck with nearby customers, reducing total travel distance.
    # 1/d^2 is more aggressive than 1/d in preferring close neighbors.
    return 1.0 / (distance_matrix ** 2 + 1e-9)
# EVOLVE-BLOCK-END
