import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x_flat, temperature=0.05):
    """
    Smooth surrogate for the min-triangle-area objective.
    Uses a soft-min to make the landscape differentiable-friendly for gradient-free methods.
    Also penalizes points that are too close to each other or outside a reasonable bound.
    """
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Soft penalty for points going far out
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm > 3.0:
        return 1e10 + (max_norm - 3.0) ** 2

    areas = _triangle_areas_vectorized(points)
    hull_area = _hull_area(points)
    
    if hull_area < 1e-12:
        return 1e10

    # Normalize areas by hull area to make scale-invariant
    norm_areas = areas / hull_area
    
    # Soft min: -log(sum(exp(-area/T))) / N
    # This approximates min(areas) as T -> 0
    scaled = -norm_areas / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = -(log_sum) * temperature
    
    # We want to maximize soft_min, so minimize -soft_min
    # Also add a small regularization to encourage spread
    # Actually, just return -soft_min
    return -soft_min


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _generate_initial_guesses(rng):
    """Generate diverse initial configurations."""
    guesses = []
    
    # 1. Regular 13-gon
    guesses.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings: 1 center, 6 inner, 6 outer
    center = np.array([[0.0, 0.0]])
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    guesses.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Two-ring structure: 5 inner, 8 outer
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    inner_pts = np.column_stack([0.6 * np.cos(inner_angles), 0.6 * np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    guesses.append(np.vstack([inner_pts, outer_pts]))
    
    # 4. Radial perturbations with multiple harmonics
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    for harmonic in [2, 3, 5]:
        radii = 1.0 + 0.1 * np.cos(harmonic * base_angles)
        pts = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
        guesses.append(pts)
    
    # 5. Random points in disk
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        guesses.append(pts)
    
    # 6. Random points in square
    for _ in range(3):
        pts = rng.uniform(-1, 1, (N, 2))
        guesses.append(pts)
    
    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Evaluate initial guesses
    guesses = _generate_initial_guesses(rng)
    for guess in guesses:
        _update_best(guess)
    
    # Phase 2: Multi-stage optimization with soft-min objective
    # Use multiple temperatures, starting high and cooling down
    temperatures = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() > deadline - 15:
            break
        
        # Try to optimize from best point found so far
        if best_points is not None:
            x0 = best_points.flatten()
            try:
                result = minimize(
                    _soft_min_objective,
                    x0,
                    args=(temp,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                _update_best(result.x.reshape(N, 2))
            except Exception:
                pass
        
        # Also try from a few other good initializations
        for guess in guesses[:3]:
            if time.time() > deadline - 10:
                break
            x0 = guess.flatten()
            try:
                result = minimize(
                    _soft_min_objective,
                    x0,
                    args=(temp,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                _update_best(result.x.reshape(N, 2))
            except Exception:
                pass
    
    # Phase 3: Local polish with hard objective (min triangle area / hull area)
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            # Use a tighter optimization with the actual score
            def _hard_objective(x_flat):
                points = x_flat.reshape(N, 2)
                if not np.all(np.isfinite(points)):
                    return 1e10
                s = _score(points)
                if s <= 0:
                    return 1e10
                return -s
            
            result = minimize(
                _hard_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 4: Additional random restarts if time permits
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation of best point
            if best_points is not None:
                perturbed = best_points + rng.normal(0, 0.1, (N, 2))
                _update_best(perturbed)
                
                if time.time() < deadline - 5:
                    try:
                        result = minimize(
                            _soft_min_objective,
                            perturbed.flatten(),
                            args=(0.01,),
                            method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                        _update_best(result.x.reshape(N, 2))
                    except Exception:
                        pass
            
            # Or from a fresh random guess
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.3, 1.0, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            _update_best(pts)
            
            if time.time() < deadline - 5:
                try:
                    result = minimize(
                        _soft_min_objective,
                        pts.flatten(),
                        args=(0.05,),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    _update_best(result.x.reshape(N, 2))
                except Exception:
                    pass
    
    # Final check
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
