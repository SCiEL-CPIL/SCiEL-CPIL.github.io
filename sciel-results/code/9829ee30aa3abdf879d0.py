"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time
import sys

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for vectorization
_triple_indices = np.array(list(itertools.combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _compute_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    # points shape (N, 2)
    # triple_indices shape (NUM_TRIPLES, 3)
    p1 = points[_triple_indices[:, 0]]
    p2 = points[_triple_indices[:, 1]]
    p3 = points[_triple_indices[:, 2]]
    
    # Area = 0.5 * | (p2 - p1) x (p3 - p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert parameters to points.
    params: [log_dists (N), angle_perts (N)]
    Uses support line representation.
    """
    n = N
    log_dists = params[:n]
    angle_perts = params[n:]
    
    # Base angles equally spaced
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = base_angles + angle_perts
    
    # Ensure angles are sorted to maintain vertex order
    # Since perturbations are small, sorting is stable
    angles = np.sort(angles)
    
    d = np.exp(log_dists)
    
    points = np.zeros((n, 2))
    for i in range(n):
        i1 = i
        i2 = (i - 1) % n
        
        a1 = np.array([np.cos(angles[i1]), np.sin(angles[i1])])
        a2 = np.array([np.cos(angles[i2]), np.sin(angles[i2])])
        
        A = np.array([a1, a2])
        b = np.array([d[i1], d[i2]])
        
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                points[i] = a1 * d[i1]
            else:
                points[i] = np.linalg.solve(A, b)
        except Exception:
            points[i] = a1 * d[i1]
            
    return points

def _soft_min_objective(params, beta):
    """
    Objective function using soft-min approximation.
    Minimizes -soft_min(areas) / hull_area
    """
    points = _params_to_points(params)
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10  # Large penalty
        
    areas = _triangle_areas(points)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, use log-sum-exp trick
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - max_area)))) - beta * max_area
    soft_min_area = -log_sum_exp / beta
    
    score = soft_min_area / hull_area
    return -score

def _exact_objective(params):
    """
    Objective function using exact min.
    Minimizes -min(areas) / hull_area
    """
    points = _params_to_points(params)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Bounds for parameters
    # log_dists: roughly around 0 (radius 1)
    # angle_perts: small perturbations to maintain order
    log_d_bounds = [(-1.0, 1.0)] * N
    angle_perts_bounds = [(-0.2, 0.2)] * N
    bounds = log_d_bounds + angle_perts_bounds
    
    best_score = -np.inf
    best_params = None
    
    # Multi-start optimization
    n_starts = 10
    rng = np.random.RandomState(42)
    
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
            
        # Initialize params
        # Start near regular polygon
        init_params = np.zeros(2 * N)
        # Add small noise
        init_params += rng.uniform(-0.1, 0.1, size=2 * N)
        
        # Annealing betas
        betas = [10, 50, 200, 1000]
        current_params = init_params.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    _soft_min_objective,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.success:
                    current_params = res.x
            except Exception:
                pass
        
        # Evaluate exact score for this candidate
        try:
            candidate_points = _params_to_points(current_params)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = current_params.copy()
        except Exception:
            pass
            
    # Final polish with Nelder-Mead on exact objective
    if best_params is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < -best_score:  # res.fun is -score
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass
            
    # Generate final points
    if best_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        best_points = _params_to_points(best_params)
        
    # Normalize to unit area
    try:
        hull = ConvexHull(best_points)
        area = hull.volume
        if area > 1e-10:
            best_points = best_points / np.sqrt(area)
    except Exception:
        pass
        
    return best_points

# EVOLVE-BLOCK-END
