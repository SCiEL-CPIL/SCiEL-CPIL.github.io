"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    # Use itertools.combinations for clarity, N=14 is small enough (364 triangles)
    # For speed in optimization, we might want a vectorized approach, but 
    # 364 triangles * 2 floats is very fast in Python/NumPy if done right.
    # Let's do a vectorized calculation for all pairs of points first?
    # Actually, just iterating combinations is fine for N=14.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, return 0 or a small number
        return 1e-10

def _score(points):
    """Calculate the score: min_triangle_area / convex_hull_area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _objective(points_flat):
    """Objective function for minimization (we want to maximize score, so minimize -score)."""
    points = points_flat.reshape(-1, 2)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    best_score = -np.inf
    best_points = None
    
    # 1. Regular N-gon (Baseline)
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    score_reg = _score(pts_reg)
    if score_reg > best_score:
        best_score = score_reg
        best_points = pts_reg.copy()
        
    # 2. Generate diverse initializations
    initializations = []
    
    # Regular polygon
    initializations.append(pts_reg.copy())
    
    # Concentric rings (2 rings: 7 inner, 7 outer)
    # Inner radius 0.5, Outer radius 1.0
    # Rotate inner ring slightly to avoid alignment
    angles_inner = 2 * np.pi * np.arange(7) / 7 + np.pi/7
    angles_outer = 2 * np.pi * np.arange(7) / 7
    pts_inner = np.column_stack((0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)))
    pts_outer = np.column_stack((1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)))
    pts_rings = np.vstack((pts_inner, pts_outer))
    initializations.append(pts_rings)
    
    # Random perturbations of regular polygon
    rng = np.random.default_rng(42)
    for i in range(20):
        perturbed = pts_reg + rng.normal(0, 0.1, size=(N, 2))
        initializations.append(perturbed)
        
    # Random points in unit circle
    for i in range(10):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(pts_rand)
        
    # 3. Optimize each initialization
    # Use Powell method which is derivative-free and robust
    # Nelder-Mead is also an option but can be slower to converge precisely
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
            
        x0 = init_pts.flatten()
        
        # Try Powell
        try:
            res = minimize(_objective, x0, method='Powell', options={'maxiter': 200, 'disp': False})
            if res.success or res.fun < -best_score:
                pts_opt = res.x.reshape(-1, 2)
                score_opt = -res.fun
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = pts_opt.copy()
        except Exception:
            pass
            
        # Try Nelder-Mead if time permits
        if time.time() < deadline - 5.0:
            try:
                res_nm = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'disp': False})
                if res_nm.fun < -best_score:
                    pts_opt = res_nm.x.reshape(-1, 2)
                    score_opt = -res_nm.fun
                    if score_opt > best_score:
                        best_score = score_opt
                        best_points = pts_opt.copy()
            except Exception:
                pass

    # 4. Final Refinement on the best point found
    if best_points is not None and time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            # Tighter tolerance, more iterations
            res_final = minimize(_objective, x_best, method='Powell', options={'maxiter': 500, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
            if res_final.fun < -best_score:
                pts_final = res_final.x.reshape(-1, 2)
                score_final = -res_final.fun
                if score_final > best_score:
                    best_score = score_final
                    best_points = pts_final.copy()
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
