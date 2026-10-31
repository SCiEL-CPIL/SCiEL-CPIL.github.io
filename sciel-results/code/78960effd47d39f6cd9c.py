"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

# Precompute all triples for N=13
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    min_area = np.inf
    for i, j, k in _TRIPLES:
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _objective(flat_points):
    """Objective function for optimization (minimize negative score)."""
    points = flat_points.reshape(N, 2)
    score = _score(points)
    return -score

def _refine(points, max_iter=200, time_limit=10):
    """Local refinement using Nelder-Mead."""
    flat = points.flatten()
    start_time = time.time()
    
    # Use Nelder-Mead for local optimization
    best_result = None
    try:
        result = minimize(
            _objective,
            flat,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if result.fun < 0:  # Valid result
            best_result = result.x.reshape(N, 2)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_result, elapsed

def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: Differential evolution for global optimization
    # Bounds: points in [0, 1]^2
    bounds = [(0, 1)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    # Run differential evolution with multiple attempts if time permits
    de_configs = [
        {'popsize': 30, 'maxiter': 100, 'tol': 1e-6, 'seed': 42},
        {'popsize': 20, 'maxiter': 150, 'tol': 1e-7, 'seed': 123},
        {'popsize': 25, 'maxiter': 120, 'tol': 1e-6, 'seed': 456},
    ]
    
    for config in de_configs:
        if time.time() > deadline - 30:  # Need time for refinement
            break
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                **config
            )
            if result.fun < best_score:
                best_score = result.fun
                best_points = result.x.reshape(N, 2)
        except Exception:
            continue
        
        if time.time() > deadline - 30:
            break
    
    # If DE didn't find anything good, use a structured initial guess
    if best_points is None or best_score > -0.01:
        # Regular polygon-like distribution
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Mix of inner and outer points
        radii = np.array([0.5] * (N//2) + [1.0] * (N - N//2))
        # Shuffle to mix
        idx = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(idx)
        radii = radii[idx]
        
        best_points = np.column_stack([
            0.5 + radii * np.cos(angles),
            0.5 + radii * np.sin(angles)
        ])
        best_score = -_score(best_points)
    
    # Phase 2: Local refinement
    if best_points is not None and time.time() < deadline - 5:
        refined, elapsed = _refine(best_points, max_iter=300, time_limit=30)
        if refined is not None:
            refined_score = _score(refined)
            if refined_score > -best_score:
                best_points = refined
                best_score = -refined_score
    
    # Phase 3: Additional refinement attempts with different perturbations
    if best_points is not None and time.time() < deadline - 5:
        np.random.seed(789)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
            # Clip to valid range
            perturbed = np.clip(perturbed, 0, 1)
            
            refined, elapsed = _refine(perturbed, max_iter=200, time_limit=20)
            if refined is not None:
                refined_score = _score(refined)
                if refined_score > -best_score:
                    best_points = refined
                    best_score = -refined_score
    
    # Final: normalize to unit area by scaling
    if best_points is not None:
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-10:
            # Scale to unit area
            scale = 1.0 / np.sqrt(hull_area)
            best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
