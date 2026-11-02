"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, LinearConstraint
import itertools
import math

N = 11

def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three of the given points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """Objective function to minimize (negative of min triangle area)."""
    points = params.reshape(N, 2)
    # Check if all points are inside the equilateral triangle
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # A point (x,y) is inside if:
    # y >= 0
    # y <= sqrt(3) * x  (below the line from (0,0) to (0.5, sqrt(3)/2))
    # y <= -sqrt(3) * (x - 1)  (below the line from (1,0) to (0.5, sqrt(3)/2))
    for i in range(N):
        x, y = points[i]
        if y < -1e-10:
            return 1e6  # Penalty
        if y > math.sqrt(3) * x + 1e-10:
            return 1e6
        if y > -math.sqrt(3) * (x - 1) + 1e-10:
            return 1e6
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    """Place 11 points in the unit equilateral triangle to maximize the minimum triangle area."""
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, math.sqrt(3) / 2)] * N
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds=bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        popsize=15,
        init='sobol',
        polish=True
    )
    
    points = result.x.reshape(N, 2)
    return points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
