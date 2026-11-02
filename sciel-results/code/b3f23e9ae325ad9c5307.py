"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 10.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _neg_score(pts_flat):
    """Objective function for minimization: -score."""
    pts = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _make_config_A(params):
    """
    Construct points from parameters: [r_inner, r_outer, angle_offset_inner, angle_offset_outer]
    Structure: 1 center point, 6 inner points, 6 outer points.
    """
    r_in, r_out, a_in, a_out = params
    pts = np.zeros((N, 2))
    # Center
    pts[0] = [0, 0]
    
    # Inner ring (6 points)
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + a_in
    pts[1:7, 0] = r_in * np.cos(angles_in)
    pts[1:7, 1] = r_in * np.sin(angles_in)
    
    # Outer ring (6 points)
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + a_out
    pts[7:13, 0] = r_out * np.cos(angles_out)
    pts[7:13, 1] = r_out * np.sin(angles_out)
    
    return pts

def _neg_score_A(params):
    """Objective for parameterized config A."""
    pts = _make_config_A(params)
    return -_score(pts)

def construct():
    entry_time = time.time()
    deadline = entry_time + (DEADLINE - SAFETY_MARGIN)
    
    best_score = 0.0
    best_points = None
    
    # Strategy:
    # 1. Explore structured symmetric configurations (concentric rings) using Differential Evolution.
    #    This avoids the high-dimensional local minima traps of free 26D search.
    # 2. Take the best structured result and refine it in the full 26D space using Nelder-Mead.
    
    # --- Phase 1: Differential Evolution on symmetric parameters ---
    # Parameters: [r_inner, r_outer, angle_offset_inner, angle_offset_outer]
    # Bounds:
    # r_inner: [0.1, 0.9] (keep away from 0 to avoid degeneracy, but allow small)
    # r_outer: [0.5, 1.5] (should be larger than inner)
    # angles: [0, pi/3] (symmetry period for hexagon)
    
    bounds = [
        (0.05, 0.9),   # r_inner
        (0.5, 1.5),    # r_outer
        (0.0, np.pi/3),# angle_inner
        (0.0, np.pi/3) # angle_outer
    ]
    
    # We can also try a 7-6 split (1 center, 7 inner, 5 outer? No, 1+6+6 is 13. 1+7+5=13. Let's stick to 1+6+6 for symmetry or 1+7+5).
    # Actually, 13 points: 1 center + 12 on rings is standard.
    # Let's also try a 13-gon structure? No, rings are better for max-min area usually.
    # Let's run DE on the 1+6+6 structure.
    
    try:
        # Use DE to find good parameters
        # popsize=15, maxiter=50 to keep it fast but effective
        # tol=1e-6
        res_de = differential_evolution(
            _neg_score_A, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6,
            polish=False # We will polish later
        )
        
        if res_de.success or res_de.fun < 0:
            best_params = res_de.x
            pts_de = _make_config_A(best_params)
            s_de = -res_de.fun
            if s_de > best_score:
                best_score = s_de
                best_points = pts_de.copy()
    except Exception:
        pass
    
    # Also try a slightly different symmetric structure: 1 center + 7 inner + 5 outer?
    # Or just a regular 13-gon?
    # Let's add a regular 13-gon as a candidate.
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    poly13 = np.column_stack([np.cos(theta), np.sin(theta)])
    s_poly = _score(poly13)
    if s_poly > best_score:
        best_score = s_poly
        best_points = poly13.copy()
        
    # If DE failed or didn't run, use a heuristic init
    if best_points is None:
        # Fallback: 1 center, 12 ring
        center = np.array([[0.0, 0.0]])
        theta12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring12 = np.column_stack([np.cos(theta12), np.sin(theta12)])
        best_points = np.vstack([center, ring12])
        best_score = _score(best_points)

    # --- Phase 2: Nelder-Mead Refinement in Full Space ---
    # Now that we have a good structured starting point, we break the symmetry
    # and let NM fine-tune the exact positions. This often yields a significant boost
    # because the symmetric solution is rarely the exact global optimum in 2D.
    
    if time.time() < deadline - 20:
        x0 = best_points.flatten()
        
        # Run NM from the best point
        try:
            res_nm = minimize(
                _neg_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < -best_score: # Improvement
                pts_nm = res_nm.x.reshape(N, 2)
                s_nm = -res_nm.fun
                if s_nm > best_score:
                    best_score = s_nm
                    best_points = pts_nm.copy()
        except Exception:
            pass
            
        # Try a few perturbations if time remains
        if time.time() < deadline - 10:
            rng = np.random.default_rng(42)
            for _ in range(3):
                if time.time() > deadline - 5: break
                x_pert = best_points.flatten() + rng.normal(0, 0.02, (2*N,))
                try:
                    res_p = minimize(
                        _neg_score, 
                        x_pert, 
                        method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
                    if res_p.fun < -best_score:
                        pts_p = res_p.x.reshape(N, 2)
                        s_p = -res_p.fun
                        if s_p > best_score:
                            best_score = s_p
                            best_points = pts_p.copy()
                except Exception:
                    pass

    # Final normalization (optional, score is scale invariant, but good for sanity)
    if best_points is not None:
        # Center and scale to unit max norm? No, score is invariant. 
        # Just ensure it's a valid numpy array.
        pass
        
    return best_points

# EVOLVE-BLOCK-END
