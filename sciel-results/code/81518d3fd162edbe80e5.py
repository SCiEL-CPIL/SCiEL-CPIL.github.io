"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

def _precompute_indices():
    """Precompute all (i,j,k) index triples for N points."""
    idxs = list(combinations(range(N), 3))
    return np.array(idxs)

_IDX = _precompute_indices()
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _min_triangle_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three points.
    Vectorized for speed.
    points: (N, 2) array
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _violation(points):
    """
    Calculate the violation of the triangle constraints.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Constraints:
    1. y >= 0
    2. y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    3. y <= -sqrt(3) * (x - 1) => sqrt(3)*(1-x) - y >= 0
    """
    x = points[:, 0]
    y = points[:, 1]
    sqrt3 = np.sqrt(3)
    
    # Distances to boundaries (negative means outside)
    # We want these to be >= 0
    d1 = y 
    d2 = sqrt3 * x - y
    d3 = sqrt3 * (1 - x) - y
    
    # Sum of squared violations for points that are outside
    # We only care about points that violate, so we take max(0, -d)
    v1 = np.maximum(0, -d1)
    v2 = np.maximum(0, -d2)
    v3 = np.maximum(0, -d3)
    
    return np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    sqrt3 = np.sqrt(3)
    eps = 1e-9
    
    # y bounds
    y = np.clip(y, eps, h - eps)
    
    # x bounds given y
    # y <= sqrt3 * x => x >= y/sqrt3
    # y <= sqrt3 * (1-x) => sqrt3*x <= sqrt3 - y => x <= 1 - y/sqrt3
    x_min = y / sqrt3
    x_max = 1 - y / sqrt3
    
    x = np.clip(x, x_min + eps, x_max - eps)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _objective_de(params):
    """
    Objective function for Differential Evolution.
    Minimizes -min_area + penalty * violation
    """
    points = params.reshape(N, 2)
    min_area = _min_triangle_area_vec(points)
    viol = _violation(points)
    # Large penalty to ensure points stay inside
    return -min_area + 100.0 * viol

def _objective_local(params):
    """
    Objective function for local refinement (Nelder-Mead/BFGS).
    Uses a softer penalty or assumes points are already close to valid.
    """
    points = params.reshape(N, 2)
    min_area = _min_triangle_area_vec(points)
    viol = _violation(points)
    # Smaller penalty for local refinement, as we expect small violations
    return -min_area + 1000.0 * viol

def _generate_initial_guesses():
    """Generate a few diverse initial guesses for DE."""
    h = np.sqrt(3) / 2
    guesses = []
    
    # 1. Vertices and centroid
    sym1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4]
    ])
    guesses.append(sym1.flatten())
    
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        pts = _clamp_all(pts)
        guesses.append(pts.flatten())
        
    return guesses

def _local_refine(points, time_limit=5.0):
    """
    Refine points using Nelder-Mead on the full vector, then coordinate descent.
    """
    points = np.array(points, dtype=float)
    points = _clamp_all(points)
    best_points = points.copy()
    best_score = _min_triangle_area_vec(points)
    
    start_time = time.time()
    
    # Stage 1: Global local refinement (Nelder-Mead on 22D)
    if time.time() - start_time < time_limit * 0.5:
        try:
            res = minimize(
                _objective_local,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_pts = res.x.reshape(N, 2)
            opt_pts = _clamp_all(opt_pts)
            score = _min_triangle_area_vec(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts
        except Exception:
            pass

    # Stage 2: Coordinate descent (Optimize each point individually)
    if time.time() - start_time < time_limit:
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            p = best_points[i].copy()
            
            def obj_single(x):
                pts = best_points.copy()
                pts[i] = _clamp_to_triangle(x)
                return -_min_triangle_area_vec(pts)
            
            try:
                res = minimize(
                    obj_single,
                    p,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                new_p = _clamp_to_triangle(res.x)
                old_p = best_points[i]
                best_points[i] = new_p
                score = _min_triangle_area_vec(best_points)
                if score > best_score:
                    best_score = score
                else:
                    # Revert if it got worse (shouldn't happen with NM if it converges to local min, but safe)
                    best_points[i] = old_p
            except Exception:
                best_points[i] = old_p

    return best_points, best_score

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Bounds for DE: x in [0, 1], y in [0, sqrt(3)/2]
    h = np.sqrt(3) / 2
    bounds = [(0, 1), (0, h)] * N
    
    # Run Differential Evolution
    # We use a moderate population size and tolerance to balance exploration/exploitation
    # time_left is used to estimate maxiter if needed, but DE has its own stopping criteria
    # We rely on the 'tol' and 'maxiter' options.
    
    try:
        # DE can be slow, so we set a reasonable maxiter. 
        # 150 iterations with popsize 15 is roughly 2250 function evaluations.
        # Each evaluation is ~1ms, so this should take a few seconds.
        # We can run it for a fixed duration by wrapping it or just setting high maxiter 
        # and letting it converge or hit time limit. 
        # However, scipy's DE doesn't have a direct time limit. 
        # We will run it with a generous maxiter and check time afterwards.
        
        # To strictly respect time, we might need to run DE in a way that allows interruption,
        # but standard scipy DE is a black box. 
        # Given 140s budget, and DE being relatively fast per iteration, 
        # a single run with maxiter=100-200 is usually safe.
        
        # Let's run DE with a moderate setting.
        de_result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False  # We will do our own polishing
        )
        
        de_points = de_result.x.reshape(N, 2)
        de_points = _clamp_all(de_points)
        de_score = _min_triangle_area_vec(de_points)
        
        if de_score > best_score:
            best_score = de_score
            best_points = de_points
            
    except Exception:
        pass

    # If DE failed or didn't produce a result, fall back to a random start
    if best_points is None:
        best_points = _generate_initial_guesses()[0].reshape(N, 2)
        best_points = _clamp_all(best_points)
        best_score = _min_triangle_area_vec(best_points)

    # Refine the best point found
    remaining_time = deadline - time.time() - 2.0  # Leave 2s for final checks
    if remaining_time > 1.0:
        try:
            refined_points, refined_score = _local_refine(best_points, time_limit=remaining_time)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
        except Exception:
            pass

    # Final sanity check: ensure points are inside
    best_points = _clamp_all(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
