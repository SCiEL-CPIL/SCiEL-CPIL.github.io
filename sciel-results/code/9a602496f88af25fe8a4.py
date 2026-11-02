import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import sys

N = 14

# Precompute triangle indices for vectorized area calculation
# C(14, 3) = 364 triangles
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """
    Compute min_triangle_area / convex_hull_area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Vectorized triangle area calculation
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def neg_score(x):
    """Objective function for minimizers: -score"""
    points = x.reshape((N, 2))
    return -compute_score(points)

def two_ring_to_points(params):
    """
    Convert 2-ring parameters to point coordinates.
    params: [r1, r2, th1, th2, n1]
    n1 is the count on the outer ring (inner has N-n1).
    """
    r1, r2, th1, th2, n1 = params
    n1 = int(round(n1))
    n1 = max(3, min(N-3, n1))  # Ensure at least 3 on each ring for valid triangles
    n2 = N - n1
    
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
        
    result = np.vstack(points)
    return result

def neg_two_ring_score(params):
    """Objective for differential evolution on 2-ring parameters."""
    try:
        points = two_ring_to_points(params)
        return -compute_score(points)
    except Exception:
        return 0.0

def three_ring_to_points(params):
    """
    Convert 3-ring parameters to point coordinates.
    params: [r1, r2, r3, th1, th2, th3, n1, n2]
    n1 outer, n2 middle, n3 = N - n1 - n2 inner.
    """
    r1, r2, r3, th1, th2, th3, n1, n2 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = N - n1 - n2
    
    # Ensure valid counts
    n1 = max(1, min(10, n1))
    n2 = max(1, min(10, n2))
    n3 = N - n1 - n2
    if n3 < 1:
        n2 = max(1, N - n1 - 1)
        n3 = N - n1 - n2
    
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
        
    result = np.vstack(points)
    return result

def neg_three_ring_score(params):
    """Objective for differential evolution on 3-ring parameters."""
    try:
        points = three_ring_to_points(params)
        return -compute_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin for 150s limit
    
    best_points = None
    best_score = -np.inf
    rng = np.random.default_rng(42)
    
    def update_best(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Phase 1: Global Search with Differential Evolution ---
    
    # 1a: Two-ring configuration (7+7 or similar splits)
    if time.time() - start_time < deadline - 60:
        bounds_2ring = [
            (0.8, 1.5),  # r1 (outer)
            (0.1, 0.8),  # r2 (inner)
            (0, 2*np.pi),  # th1
            (0, 2*np.pi),  # th2
            (4, 10),  # n1 (outer count)
        ]
        
        for seed in [42, 142, 242]:
            if time.time() - start_time > deadline - 60:
                break
            try:
                res = differential_evolution(
                    neg_two_ring_score,
                    bounds_2ring,
                    seed=seed,
                    maxiter=80,
                    popsize=20,
                    mutation=(0.3, 1.2),
                    recombination=0.8,
                    tol=1e-7,
                    polish=False,
                    init='latinhypercube'
                )
                if res.fun < 0:
                    points = two_ring_to_points(res.x)
                    update_best(points)
            except Exception:
                pass

    # 1b: Three-ring configuration
    if time.time() - start_time < deadline - 30:
        bounds_3ring = [
            (0.8, 1.5),  # r1 (outer)
            (0.3, 1.0),  # r2 (middle)
            (0.0, 0.5),  # r3 (inner)
            (0, 2*np.pi),  # th1
            (0, 2*np.pi),  # th2
            (0, 2*np.pi),  # th3
            (3, 9),  # n1
            (2, 8),  # n2
        ]
        
        for seed in [42, 142]:
            if time.time() - start_time > deadline - 30:
                break
            try:
                res = differential_evolution(
                    neg_three_ring_score,
                    bounds_3ring,
                    seed=seed,
                    maxiter=50,
                    popsize=15,
                    mutation=(0.3, 1.2),
                    recombination=0.8,
                    tol=1e-6,
                    polish=False,
                    init='latinhypercube'
                )
                if res.fun < 0:
                    points = three_ring_to_points(res.x)
                    update_best(points)
            except Exception:
                pass

    # Fallback if no valid points found
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        update_best(best_points)
        
    # --- Phase 2: Local Refinement on Coordinates ---
    
    candidates_to_refine = [best_points]
    
    # Add some structured perturbations
    for scale in [0.03, 0.06, 0.1]:
        if time.time() - start_time > deadline - 15:
            break
        perturbed = best_points + rng.normal(0, scale, (N, 2))
        candidates_to_refine.append(perturbed)
        
    # Add a standard 7+7 two-ring configuration
    a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a_in = a_out + np.pi/7
    two_ring = np.vstack((
        np.column_stack((1.0*np.cos(a_out), 1.0*np.sin(a_out))),
        np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
    ))
    candidates_to_refine.append(two_ring)
    
    # Add a 6+6+2 three-ring configuration
    a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    a2 = a1 + np.pi/6
    a3 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    three_ring = np.vstack((
        np.column_stack((1.0*np.cos(a1), 1.0*np.sin(a1))),
        np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
        np.column_stack((0.2*np.cos(a3), 0.2*np.sin(a3)))
    ))
    candidates_to_refine.append(three_ring)
    
    for cand in candidates_to_refine:
        if time.time() - start_time > deadline - 5:
            break
            
        x0 = cand.flatten()
        
        try:
            # Powell is robust for non-smooth objectives (min of areas)
            res = minimize(
                neg_score, 
                x0, 
                method='Powell', 
                options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if res.fun < -best_score:
                best_points = res.x.reshape((N, 2))
                best_score = -res.fun
                
        except Exception:
            pass

    # --- Phase 3: Final fine-tuning with Nelder-Mead ---
    if time.time() - start_time < deadline - 2:
        try:
            res = minimize(
                neg_score, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < -best_score:
                best_points = res.x.reshape((N, 2))
                best_score = -res.fun
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
