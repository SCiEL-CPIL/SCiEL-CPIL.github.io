"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Sort by x, then y
    points_sorted = sorted(points, key=lambda p: (p[0], p[1]))
    
    # Build lower hull
    lower = []
    for p in points_sorted:
        while len(lower) >= 2 and np.cross(
            np.array(lower[-1]) - np.array(lower[-2]), 
            np.array(p) - np.array(lower[-1])) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points_sorted):
        while len(upper) >= 2 and np.cross(
            np.array(upper[-1]) - np.array(upper[-2]), 
            np.array(p) - np.array(upper[-1])) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate, removing duplicates of last point
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Compute area using shoelace formula
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)


def _score(points):
    """Score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def construct():
    """
    Construct 13 points in the unit square to maximize the minimum triangle area
    divided by the convex hull area.
    
    Uses a multi-start differential evolution approach with a structured initial
    guess based on a regular polygon arrangement.
    """
    n = N
    
    # Create a good initial guess: points on a circle (regular polygon)
    # centered at (0.5, 0.5) with radius 0.4
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 0.4
    center = np.array([0.5, 0.5])
    initial_guess = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Flatten to 1D array for optimization
    x0 = initial_guess.flatten()
    
    # Bounds: all points in [0, 1] x [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    def _objective(x):
        points = x.reshape(n, 2)
        return -_score(points)  # Negate for minimization
    
    # Run differential evolution
    best_result = differential_evolution(
        _objective,
        bounds=bounds,
        seed=42,
        maxiter=200,
        tol=1e-8,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.8,
        polish=True,
        init='sobol'
    )
    
    best_points = best_result.x.reshape(n, 2)
    best_score = -best_result.fun
    
    # Try a few more local refinements with different starting points
    for seed in [1, 2, 3, 4, 5]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.default_rng(seed).uniform(0, 0.5)
        radii = 0.35 + np.random.default_rng(seed).uniform(0, 0.1, n)
        center = np.array([0.5, 0.5])
        guess = center + radii * np.column_stack([np.cos(angles), np.sin(angles)])
        # Clip to [0, 1]
        guess = np.clip(guess, 0.0, 1.0)
        
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=seed,
            maxiter=100,
            tol=1e-8,
            popsize=10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube'
        )
        
        if -result.fun > best_score:
            best_score = -result.fun
            best_points = result.x.reshape(n, 2)
    
    return best_points
# EVOLVE-BLOCK-END
