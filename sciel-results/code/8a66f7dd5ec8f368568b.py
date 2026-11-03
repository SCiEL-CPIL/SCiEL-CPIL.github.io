"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                cross = dx12 * (y3 - y1) - dy12 * (x3 - x1)
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian in the equilateral triangle."""
    a, b, c = bary
    x = a + 0.5 * b
    y = (np.sqrt(3) / 2) * b
    return x, y


def _in_triangle(x, y, eps=1e-6):
    """Check if point is inside the equilateral triangle."""
    if y < -eps or y > np.sqrt(3) / 2 + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    # Check below the top edge
    if x < 0.5:
        # Left side: y <= sqrt(3) * x
        if y > np.sqrt(3) * x + eps:
            return False
    else:
        # Right side: y <= sqrt(3) * (1 - x)
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
    return True


def _objective_flat(flat_params):
    """Objective function for differential evolution (minimize negative min area)."""
    points = []
    for i in range(N):
        u = flat_params[2 * i]
        v = flat_params[2 * i + 1]
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        if not _in_triangle(x, y, eps=0.01):
            return -1e-10  # Heavy penalty for out-of-bounds
        points.append([x, y])
    
    min_area = _min_triangle_area(np.array(points))
    return -min_area


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial good starting point: distribute points roughly evenly
    # Use a grid-like initial distribution in barycentric coordinates
    initial_points = []
    # Corners and edge midpoints and center
    barycentric_init = [
        [0.1, 0.1, 0.8],   # Near bottom-left
        [0.1, 0.8, 0.1],   # Near bottom-right  
        [0.8, 0.1, 0.1],   # Near top
        [0.15, 0.15, 0.7], # Bottom edge left
        [0.15, 0.7, 0.15], # Bottom edge right
        [0.7, 0.15, 0.15], # Top edge left
        [0.7, 0.15, 0.15], # Top edge right
        [0.333, 0.333, 0.334], # Center
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2],
    ]
    
    for b in barycentric_init:
        x, y = _barycentric_to_cartesian(b)
        initial_points.append([x, y])
    
    initial_flat = []
    for p in initial_points:
        # Convert back to u, v where x = u + 0.5*v, y = sqrt(3)/2 * v
        v = 2 * p[1] / np.sqrt(3)
        u = p[0] - 0.5 * v
        initial_flat.extend([u, v])
    
    # Define bounds for u and v
    # v ranges from 0 to 1
    # u ranges such that point is in triangle: for given v, u must satisfy constraints
    # Simplified bounds: u in [0, 1], v in [0, 1], with penalty for out-of-bounds
    bounds = [(0, 1), (0, 1)] * N
    
    # Run differential evolution
    result = differential_evolution(
        _objective_flat,
        bounds,
        x0=initial_flat,
        maxiter=200,
        popsize=20,
        tol=1e-6,
        mutation=(0.3, 1.0),
        recombination=0.8,
        polish=True,
        seed=42,
        callback=lambda x, convergence: None
    )
    
    # Extract best solution
    if result.success or result.fun < 0:
        best_flat = result.x
        points = []
        for i in range(N):
            u = best_flat[2 * i]
            v = best_flat[2 * i + 1]
            x = u + 0.5 * v
            y = (np.sqrt(3) / 2) * v
            points.append([x, y])
        points = np.array(points)
    else:
        points = np.array(initial_points)
    
    return points
# EVOLVE-BLOCK-END
