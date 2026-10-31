"""Optimized 13-point Heilbronn problem solver."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

# Precompute all triples for 13 points
_TRIPLES = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLES.shape[0]

def _compute_min_triangle_area(points):
    """Compute minimum triangle area over all triples. points shape (13,2)."""
    # points: (13, 2)
    i, j, k = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
    v1 = points[j] - points[i]  # (286, 2)
    v2 = points[k] - points[i]  # (286, 2)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _compute_convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 1.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _compute_convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _random_initial(seed):
    """Generate a random initial point set."""
    rng = np.random.RandomState(seed)
    return rng.rand(N, 2)

def _nelder_mead_optimize(points, max_iter=2000, timeout=5.0):
    """Nelder-Mead optimization on flattened point array."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 2)
        # We want to maximize score, so minimize negative score
        return -_score(pts)
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return points  # Return original if timeout
    return result.x.reshape(N, 2)

def _differential_evolution(points_list, popsize=15, maxiter=30, seed=42, timeout=10.0):
    """Differential evolution over a population of point sets."""
    from scipy.optimize import differential_evolution
    
    start_time = time.time()
    
    # Each point has 2 coordinates, so 26 dimensions
    # We'll normalize to unit square for the DE bounds
    bounds = [(0, 1)] * (2 * N)
    
    def objective(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    # Run DE with limited iterations
    result = differential_evolution(
        objective, bounds,
        popsize=popsize, maxiter=maxiter,
        seed=seed, tol=1e-8,
        polish=False,  # We'll do our own polishing
        init='sobol'
    )
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return points_list[0]  # Return best from initial population
    
    return result.x.reshape(N, 2)

def _coordinate_descent_polish(points, max_steps=500, timeout=3.0):
    """Adaptive coordinate descent to polish individual coordinates."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    step_size = 0.05
    
    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try moving in both directions
                for direction in [1, -1]:
                    trial = best_points.copy()
                    trial[i, coord] += direction * step_size
                    # Keep within [0, 1]
                    trial[i, coord] = np.clip(trial[i, coord], 0, 1)
                    
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_points = trial
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_points

def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    total_start = time.time()
    deadline = total_start + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution for global exploration
    # Use a moderate population and iterations
    de_start = time.time()
    try:
        # Run DE with limited budget
        de_points = _differential_evolution(
            [_random_initial(42)],
            popsize=20, maxiter=40, seed=42,
            timeout=min(60.0, max(5.0, deadline - time.time() - 30.0))
        )
        de_score = _score(de_points)
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
    except Exception:
        pass
    
    elapsed = time.time() - total_start
    if elapsed > deadline - 40.0:
        # Not enough time for polishing
        if best_points is not None:
            return best_points
        return _random_initial(42)
    
    # Phase 2: Nelder-Mead polishing from DE result
    if best_points is not None:
        nm_start = time.time()
        try:
            nm_points = _nelder_mead_optimize(
                best_points,
                max_iter=3000,
                timeout=min(30.0, max(5.0, deadline - time.time() - 20.0))
            )
            nm_score = _score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
        except Exception:
            pass
    
    elapsed = time.time() - total_start
    if elapsed > deadline - 20.0:
        if best_points is not None:
            return best_points
        return _random_initial(42)
    
    # Phase 3: Coordinate descent polishing
    if best_points is not None:
        cd_start = time.time()
        try:
            cd_points = _coordinate_descent_polish(
                best_points,
                max_steps=1000,
                timeout=min(15.0, max(3.0, deadline - time.time() - 5.0))
            )
            cd_score = _score(cd_points)
            if cd_score > best_score:
                best_score = cd_score
                best_points = cd_points.copy()
        except Exception:
            pass
    
    # Phase 4: Multi-start refinement with random restarts
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0 and best_points is not None:
        rng = np.random.RandomState(12345)
        n_restarts = min(5, int(remaining / 15.0))
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            # Perturb best solution
            perturbed = best_points + rng.randn(N, 2) * 0.1
            perturbed = np.clip(perturbed, 0, 1)
            
            try:
                # Quick Nelder-Mead from perturbed start
                nm_pert = _nelder_mead_optimize(
                    perturbed,
                    max_iter=1000,
                    timeout=min(10.0, max(2.0, deadline - time.time() - 5.0))
                )
                nm_pert_score = _score(nm_pert)
                if nm_pert_score > best_score:
                    best_score = nm_pert_score
                    best_points = nm_pert.copy()
            except Exception:
                pass
    
    if best_points is not None:
        return best_points
    return _random_initial(42)

# EVOLVE-BLOCK-END
