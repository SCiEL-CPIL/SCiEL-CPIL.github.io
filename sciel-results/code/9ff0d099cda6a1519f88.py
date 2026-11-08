"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # A regular n-gon inscribed in a unit circle provides a highly symmetric
    # distribution of points. For maximizing the minimum area of triangles
    # formed by subsets of points, regular polygons are often optimal or
    # near-optimal due to the uniformity of distances and areas.
    # The task normalizes by the convex hull area, so the absolute size
    # of the polygon doesn't matter, only the relative geometry.
    
    n = N
    # Generate angles for the vertices of a regular n-gon
    # endpoint=False ensures we don't duplicate the first point
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Coordinates on the unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    
    points = np.column_stack((x, y))
    
    return points
# EVOLVE-BLOCK-END
