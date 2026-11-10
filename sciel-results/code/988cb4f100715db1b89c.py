"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Use a steeper power of the inverse distance to emphasize local edges.
    # This acts as increasing the heuristic weight (beta) in standard ACO.
    # For random uniform points, the optimal tour is very local, so a higher
    # bias towards nearest neighbors helps ants avoid crossing edges and
    # find shorter tours faster.
    d = distance_matrix + 1e-9
    return 1.0 / (d * d)
# EVOLVE-BLOCK-END
