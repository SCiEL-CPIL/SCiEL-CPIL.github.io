"""Heilbronn convex regions – 13 points maximizing min triangle area / hull area."""
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points.
    Returns a 1-D array of shape (C(n,3),) with 0.5*|cross|."""
    n = len(points)
    areas = np.empty(len(list(combinations(range(n), 3))))
    idx = 0
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas[idx] = 0.5 * abs(cross)
        idx += 1
    return areas


def _convex_hull_area(points):
    """Area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # in 2D, volume == area
    except Exception:
        return 0.0


def _score(points):
    """Minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(points)
    return float(np.min(areas)) / hull_area


def _soft_min_triangle_area(points, beta=100.0):
    """Smooth surrogate: soft-min of triangle areas via log-sum-exp."""
    areas = _triangle_areas(points)
    # Avoid log of zero
    areas = np.maximum(areas, 1e-12)
    log_sum = np.log(np.sum(np.exp(-beta * areas)))
    return -np.log(np.sum(np.exp(-beta * areas))) / beta


def _objective(params, beta=100.0, penalty_weight=100.0):
    """Objective to minimize: -soft_min_area + penalty for points outside unit circle."""
    points = params.reshape(-1, 2)
    # Penalty for points outside unit circle
    dists = np.sum(points ** 2, axis=1)
    penalty = penalty_weight * np.sum(np.maximum(0, dists - 1.0))
    sm = _soft_min_triangle_area(points, beta)
    return -sm + penalty


def _local_refine(points, max_iter=200, tol=1e-10):
    """Iterative local refinement: move each point slightly to increase min area."""
    points = points.copy()
    n = len(points)
    triples = list(combinations(range(n), 3))
    
    for iteration in range(max_iter):
        areas = _triangle_areas(points)
        min_area = np.min(areas)
        min_idx = int(np.argmin(areas))
        
        # Find the triple with minimum area
        tri = triples[min_idx]
        i, j, k = tri
        
        # Try to move one of the three points to increase the area
        best_improvement = 0
        best_move = None
        best_point_idx = None
        
        for p_idx in [i, j, k]:
            # Try small perturbations
            for dx in np.linspace(-0.01, 0.01, 21):
                for dy in np.linspace(-0.01, 0.01, 21):
                    new_points = points.copy()
                    new_points[p_idx, 0] += dx
                    new_points[p_idx, 1] += dy
                    # Check if still in unit circle
                    if np.sum(new_points[p_idx] ** 2) > 1.0:
                        continue
                    new_areas = _triangle_areas(new_points)
                    new_min = np.min(new_areas)
                    improvement = new_min - min_area
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (dx, dy)
                        best_point_idx = p_idx
        
        if best_move is not None and best_improvement > tol:
            points[best_point_idx, 0] += best_move[0]
            points[best_point_idx, 1] += best_move[1]
        else:
            break
    
    return points


def _generate_initial_points(seed=42):
    """Generate initial points using a structured approach."""
    rng = np.random.RandomState(seed)
    
    # Start with points on a circle (regular polygon) and add some interior points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Add some random perturbation
    points += rng.uniform(-0.05, 0.05, size=(N, 2))
    
    # Normalize to be within unit circle
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    points = points / np.maximum(norms, 1e-10)[:, np.newaxis]
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    best_points = None
    best_score = -np.inf
    
    seeds = [42, 123, 777, 2024, 13]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        points = _generate_initial_points(seed)
        
        # Local refinement
        points = _local_refine(points, max_iter=100, tol=1e-10)
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If we have time, try a more exhaustive local search
    if best_points is not None and time.time() < deadline - 5.0:
        # Further refine with more iterations
        best_points = _local_refine(best_points, max_iter=300, tol=1e-11)
        best_score = _score(best_points)
    
    # Final check: ensure points are valid
    if best_points is None:
        # Fallback: random points
        rng = np.random.RandomState(0)
        best_points = rng.uniform(-1, 1, size=(N, 2))
        # Project to unit circle
        norms = np.sqrt(np.sum(best_points ** 2, axis=1))
        best_points = best_points / np.maximum(norms, 1e-10)[:, np.newaxis]
    
    return best_points

# EVOLVE-BLOCK-END
