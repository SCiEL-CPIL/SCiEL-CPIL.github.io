"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    # Use a sharper heuristic (1/d^2.5) to favor short edges strongly.
    # Additionally, slightly penalize edges that return to the depot (column 0)
    # to encourage ants to keep visiting customers until the truck is fuller,
    # reducing the number of trips and total distance.
    d = distance_matrix + 1e-9
    eta = 1.0 / (d ** 2.5)
    
    # Penalize returning to depot (j=0) for non-depot nodes (i!=0)
    # This makes it less attractive to return early.
    eta[1:, 0] *= 0.8
    
    return eta
# EVOLVE-BLOCK-END
