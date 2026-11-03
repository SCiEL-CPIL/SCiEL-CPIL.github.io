"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Place points at the vertices of a regular 13-gon.
    # This configuration maximizes the minimum distance between points
    # and provides a high ratio of minimum triangle area to convex hull area.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points
# EVOLVE-BLOCK-END
