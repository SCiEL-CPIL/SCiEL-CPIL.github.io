"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Generate vertices of a regular N-gon inscribed in the unit circle.
    # This configuration maximizes the minimum distance between points on the boundary
    # and provides a highly symmetric distribution, which typically maximizes
    # the minimum triangle area for a fixed convex hull area.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points
# EVOLVE-BLOCK-END
