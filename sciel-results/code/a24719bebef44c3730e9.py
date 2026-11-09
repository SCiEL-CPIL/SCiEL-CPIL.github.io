import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        # Check for degenerate hull
        if np.linalg.matrix_rank(points - points[0]) < 2:
            return 0.0
            
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _initial_points_heuristic():
    """
    Generate initial points using a nested ring heuristic.
    1 center, 5 inner (radius 0.4), 8 outer (radius 1.0).
    Angular offsets are chosen to interleave rings.
    """
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Inner ring: 5 points
    r_inner = 0.4
    angles_inner = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    xs = r_inner * np.cos(angles_inner)
    ys = r_inner * np.sin(angles_inner)
    points.extend(zip(xs, ys))
    
    # Outer ring: 8 points
    r_outer = 1.0
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    xs = r_outer * np.cos(angles_outer)
    ys = r_outer * np.sin(angles_outer)
    points.extend(zip(xs, ys))
    
    return np.array(points)

def _objective_flat(params):
    """
    Objective function for minimization (returns negative score).
    params: flat array of shape (28,)
    """
    points = params.reshape(-1, 2)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess
    x0 = _initial_points_heuristic().flatten()
    
    best_points = x0.copy()
    best_score = _score(best_points)
    
    # Try a few different initializations to avoid local minima
    # 1. The nested ring heuristic
    # 2. A regular 14-gon (flattened)
    # 3. A slightly perturbed nested ring
    
    initializations = []
    
    # Init 1: Nested rings
    initializations.append(_initial_points_heuristic().flatten())
    
    # Init 2: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    init2 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    initializations.append(init2)
    
    # Init 3: Perturbed nested rings
    init3 = _initial_points_heuristic() + np.random.normal(0, 0.05, (14, 2))
    initializations.append(init3.flatten())
    
    # Init 4: Two rings, 7 and 7
    points4 = []
    r1 = 0.5
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    points4.extend(zip(r1 * np.cos(angles1), r1 * np.sin(angles1)))
    r2 = 1.0
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    points4.extend(zip(r2 * np.cos(angles2), r2 * np.sin(angles2)))
    initializations.append(np.array(points4).flatten())
    
    for i, x0 in enumerate(initializations):
        if time.time() > deadline - 20:
            break
            
        # Use Powell's method for global-ish search in non-smooth landscape
        # Powell is robust for this kind of problem
        try:
            res = minimize(
                _objective_flat,
                x0,
                method='Powell',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                    'maxfev': 1000
                }
            )
            
            if time.time() > deadline - 5:
                break
                
            candidate_points = res.x.reshape(-1, 2)
            candidate_score = -res.fun
            
            # Verify score
            verified_score = _score(candidate_points)
            
            if verified_score > best_score:
                best_score = verified_score
                best_points = candidate_points.copy()
                
        except Exception:
            pass
            
    # If we still have time, do a finer polish with Nelder-Mead on the best result
    if time.time() < deadline - 10 and best_score > 0:
        try:
            res = minimize(
                _objective_flat,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if time.time() < deadline - 2:
                candidate_points = res.x.reshape(-1, 2)
                candidate_score = -res.fun
                verified_score = _score(candidate_points)
                
                if verified_score > best_score:
                    best_score = verified_score
                    best_points = candidate_points.copy()
        except Exception:
            pass

    # Fallback
    if best_score < 1e-6:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
