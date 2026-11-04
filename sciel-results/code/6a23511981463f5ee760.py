"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import math


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
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
    """Compute area of convex hull of points."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points (approximate)
        # This shouldn't happen for valid point sets
        return 1.0


def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _negative_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)


def _random_initial_point():
    """Generate a random initial point configuration."""
    rng = random.Random(N * 7 + 42)
    # Start with points roughly in a square, then optimize
    points = np.array([[rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1] for _ in range(N)])
    return points


def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    best_points = None
    best_score = -1.0
    
    # Try multiple random starts with local optimization
    for attempt in range(30):
        rng = random.Random(N * 13 + attempt * 31 + 7)
        
        # Initialize with points in a regular pattern perturbed by noise
        # Use a grid-like arrangement as starting point
        base_points = []
        # Place points in a rough 4x4 grid pattern (we have 14 points)
        # This gives a reasonable initial spread
        cols = 4
        rows = 4
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                x = (c + 0.5) / cols + rng.gauss(0, 0.05)
                y = (r + 0.5) / rows + rng.gauss(0, 0.05)
                base_points.append([x, y])
                idx += 1
            if idx >= N:
                break
        
        points = np.array(base_points[:N])
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _negative_score,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            opt_points = result.x.reshape(-1, 2)
            opt_score = -result.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    # If optimization failed, fall back to a structured arrangement
    if best_points is None or best_score <= 0:
        # Use a regular polygon-like arrangement
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N) * 0.5
        # Alternate between two radii for better distribution
        for i in range(N):
            radii[i] = 0.4 + 0.1 * (i % 2)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        best_points += 0.5  # Center at (0.5, 0.5)
    
    return best_points
# EVOLVE-BLOCK-END
