"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: compute area via shoelace on convex hull vertices
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            n = len(vertices)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
            return abs(area) / 2.0
        except Exception:
            return 0.0


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Triangle area = 0.5 * |cross product|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _soft_min_triangle_area(points, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta
    return soft_min


def _objective_soft(x, beta, hull_area):
    """Soft objective: soft_min_triangle_area / hull_area (to minimize, we maximize, so negate)."""
    points = x.reshape(-1, 2)
    sm = _soft_min_triangle_area(points, beta)
    # We want to maximize sm/hull_area, so minimize -sm/hull_area
    return -sm / hull_area


def _objective_hard(x, hull_area):
    """Hard objective: min_triangle_area / hull_area (negated for minimization)."""
    points = x.reshape(-1, 2)
    min_tri = _min_triangle_area(points)
    return -min_tri / hull_area


def _normalize(points):
    """Normalize points to have unit area convex hull."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return points
    # Scale points so that convex hull area becomes 1
    # Area scales as scale^2, so scale by 1/sqrt(area)
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale


def _initial_point_set(rng):
    """Generate an initial point set: regular polygon + center perturbations."""
    # Start with a regular 14-gon on a circle, then add some randomness
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.3)
    radii = np.ones(N) * (1.0 + rng.uniform(-0.1, 0.1, size=N))
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Add small random perturbations
    points += rng.uniform(-0.05, 0.05, size=(N, 2))
    return points


def construct():
    start_time = time.time()
    rng = np.random.RandomState(N)
    
    best_points = None
    best_score = -1.0
    
    # Generate multiple initial points and optimize each
    n_initials = 5
    
    for init_idx in range(n_initials):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Generate initial point set
        if init_idx == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif init_idx == 1:
            # Slightly perturbed regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.1, size=N)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif init_idx == 2:
            # Random points in unit circle
            angles = rng.uniform(0, 2 * np.pi, size=N)
            radii = np.sqrt(rng.uniform(0.3, 1.0, size=N))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif init_idx == 3:
            # Regular polygon with alternating radii
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.array([1.0 if i % 2 == 0 else 0.95 for i in range(N)])
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Random
            points = rng.uniform(-1, 1, size=(N, 2))
        
        # Normalize to unit hull area
        points = _normalize(points)
        
        # Compute initial score
        init_score = _score(points)
        if init_score > best_score:
            best_score = init_score
            best_points = points.copy()
        
        # Phase 1: Soft optimization with increasing beta (annealing)
        betas = [1.0, 5.0, 20.0, 50.0, 100.0]
        current_points = points.copy()
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            hull_area = _convex_hull_area(current_points)
            if hull_area <= 1e-12:
                break
            
            x0 = current_points.flatten()
            
            try:
                result = minimize(
                    _objective_soft,
                    x0,
                    args=(beta, hull_area),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.fun < _objective_soft(x0, beta, hull_area):
                    current_points = result.x.reshape(-1, 2)
                    current_points = _normalize(current_points)
            except Exception:
                pass
            
            # Update best
            curr_score = _score(current_points)
            if curr_score > best_score:
                best_score = curr_score
                best_points = current_points.copy()
        
        # Phase 2: Hard optimization (final polish)
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        hull_area = _convex_hull_area(current_points)
        if hull_area > 1e-12:
            x0 = current_points.flatten()
            try:
                result = minimize(
                    _objective_hard,
                    x0,
                    args=(hull_area,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < _objective_hard(x0, hull_area):
                    current_points = result.x.reshape(-1, 2)
                    current_points = _normalize(current_points)
            except Exception:
                pass
            
            curr_score = _score(current_points)
            if curr_score > best_score:
                best_score = curr_score
                best_points = current_points.copy()
        
        # Additional local search with Nelder-Mead on hard objective
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-12:
            x0 = best_points.flatten()
            try:
                result = minimize(
                    _objective_hard,
                    x0,
                    args=(hull_area,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if result.fun < _objective_hard(x0, hull_area):
                    candidate = result.x.reshape(-1, 2)
                    candidate = _normalize(candidate)
                    cand_score = _score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = candidate.copy()
            except Exception:
                pass
    
    # Final check: ensure we have a valid result
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize final result to unit hull area
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
