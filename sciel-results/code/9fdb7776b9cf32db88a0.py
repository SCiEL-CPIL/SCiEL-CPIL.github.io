"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools


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
    hull = ConvexHull(points)
    return hull.volumes  # For 2D, volumes is actually area


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _initial_points_disk(rng):
    """Generate 14 points on a unit disk with some structure."""
    points = []
    # Place some points on a circle
    n_circle = 8
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle + rng.uniform(-0.1, 0.1)
        r = 1.0 + rng.uniform(-0.05, 0.05)
        points.append([r * np.cos(angle), r * np.sin(angle)])
    # Place remaining points inside
    for i in range(N - n_circle):
        r = rng.uniform(0.3, 0.9)
        angle = rng.uniform(0, 2 * np.pi)
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _initial_points_grid(rng):
    """Generate 14 points in a grid-like pattern with jitter."""
    points = []
    # 4x4 grid minus corners, plus center
    for i in range(4):
        for j in range(4):
            if (i, j) in [(0,0), (0,3), (3,0), (3,3)]:
                continue
            x = (i - 1.5) / 2.0 + rng.uniform(-0.05, 0.05)
            y = (j - 1.5) / 2.0 + rng.uniform(-0.05, 0.05)
            points.append([x, y])
    # That gives 12 points, add 2 more
    points.append([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
    points.append([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
    return np.array(points)


def _initial_points_hexagonal(rng):
    """Generate points in a hexagonal packing pattern."""
    points = []
    # Hexagonal lattice with some perturbation
    dx = 0.5
    dy = dx * np.sqrt(3) / 2
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * dx + (j % 2) * dx / 2 + rng.uniform(-0.03, 0.03)
            y = j * dy + rng.uniform(-0.03, 0.03)
            if x*x + y*y < 1.0:
                points.append([x, y])
    # Ensure we have exactly N points
    while len(points) < N:
        r = rng.uniform(0, 1)
        angle = rng.uniform(0, 2*np.pi)
        points.append([r*np.cos(angle), r*np.sin(angle)])
    points = points[:N]
    return np.array(points)


def _optimize_points(points, max_time=30):
    """Optimize points using Nelder-Mead to maximize min triangle area / hull area."""
    start_time = time.time()
    
    def neg_score(params):
        pts = params.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    
    result = minimize(
        neg_score,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    if elapsed > max_time:
        return points  # Return original if timeout
    
    return result.x.reshape(-1, 2)


def _refine_with_local_search(points, max_time=20):
    """Local search refinement by moving points slightly."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    n = len(best_points)
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            # Try small perturbations
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    trial = best_points.copy()
                    trial[i][0] += dx
                    trial[i][1] += dy
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_points = trial
                        best_score = trial_score
                        improved = True
        if not improved:
            break
    
    return best_points


def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    import random
    rng = random.Random(N)
    
    best_points = None
    best_score = -1
    
    # Try multiple initializations and optimization strategies
    initializers = [
        _initial_points_disk,
        _initial_points_grid,
        _initial_points_hexagonal
    ]
    
    for init_func in initializers:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            points = init_func(rng)
            
            # Optimize with Nelder-Mead
            if time.time() < deadline - 20:
                points = _optimize_points(points, max_time=15)
            
            # Local search refinement
            if time.time() < deadline - 10:
                points = _refine_with_local_search(points, max_time=10)
            
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Final Nelder-Mead polish on the best
    if best_points is not None and time.time() < deadline - 5:
        def neg_score(params):
            pts = params.reshape(-1, 2)
            return -_score(pts)
        
        x0 = best_points.flatten()
        result = minimize(
            neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        
        final_points = result.x.reshape(-1, 2)
        final_score = _score(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    if best_points is None:
        # Fallback: random points
        best_points = np.array([[rng.random(), rng.random()] for _ in range(N)])
    
    return best_points
# EVOLVE-BLOCK-END
