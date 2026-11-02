import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Precompute all triangle indices for vectorized area computation
    tri_indices = np.array(list(combinations(range(N), 3)))
    n_tris = len(tri_indices)
    
    def _all_triangle_areas(pts):
        """Vectorized computation of all triangle areas."""
        i, j, k = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def _hull_area_fast(pts):
        """Compute convex hull area quickly."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def _score(pts):
        """Compute min triangle area / hull area."""
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return 0.0
        areas = _all_triangle_areas(pts)
        return np.min(areas) / hull_area
    
    def _objective_hard(flat_pts):
        """Hard objective: negative of the score."""
        pts = flat_pts.reshape(N, 2)
        s = _score(pts)
        return -s

    # --- Structural Parameterization: Two Concentric Heptagons ---
    
    def _params_to_points(params):
        """
        Convert 18 structural parameters to 14 Cartesian points.
        params[0:7] = angles for outer ring (heptagon 1)
        params[7:14] = angles for inner ring (heptagon 2)
        params[14] = radius ratio (inner/outer)
        params[15] = global rotation offset
        params[16] = radial modulation amplitude for outer
        params[17] = radial modulation amplitude for inner
        """
        # Extract parameters
        angles_outer = params[0:7]
        angles_inner = params[7:14]
        ratio = params[14]
        rotation = params[15]
        mod_outer = params[16]
        mod_inner = params[17]
        
        # Base radii
        R_outer = 1.0
        R_inner = ratio
        
        # Apply radial modulation to break perfect circularity if needed
        # Simple cosine modulation based on angle index to maintain convexity roughly
        # For simplicity in this phase, we keep them circular but allow angle jitter
        # The angles themselves are the main degrees of freedom
        
        # Generate points
        pts = np.zeros((N, 2))
        
        # Outer ring (indices 0-6)
        for i in range(7):
            theta = angles_outer[i] + rotation
            r = R_outer * (1.0 + mod_outer * np.cos(2 * theta)) # Slight ovality
            pts[i] = [r * np.cos(theta), r * np.sin(theta)]
            
        # Inner ring (indices 7-13)
        for i in range(7):
            theta = angles_inner[i] + rotation
            r = R_inner * (1.0 + mod_inner * np.cos(2 * theta))
            pts[7 + i] = [r * np.cos(theta), r * np.sin(theta)]
            
        return pts

    def _objective_structural(params):
        """Objective for the structural parameters."""
        pts = _params_to_points(params)
        return _objective_hard(pts.flatten())

    def _nm(flat_pts, maxiter=2000):
        """Nelder-Mead optimization."""
        try:
            res = minimize(_objective_hard, flat_pts, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            if res.fun < 0:
                return -res.fun, res.x
        except Exception:
            pass
        return 0.0, flat_pts
    
    def _powell(flat_pts, maxiter=3000):
        """Powell optimization."""
        try:
            res = minimize(_objective_hard, flat_pts, method='Powell',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False})
            if res.fun < 0:
                return -res.fun, res.x
        except Exception:
            pass
        return 0.0, flat_pts

    def _coordinate_descent(pts, n_iters=50, step=0.001):
        """Simple coordinate descent refinement."""
        best = pts.copy()
        best_score = _score(best)
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for coord in range(2):
                    for direction in [-1, 1]:
                        trial = best.copy()
                        trial[i, coord] += direction * step
                        s = _score(trial)
                        if s > best_score:
                            best = trial
                            best_score = s
                            improved = True
            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break
        return best_score, best

    # --- Phase 1: Optimize Structural Parameters ---
    
    # Initial guess: Regular 14-gon split into two heptagons
    # Outer heptagon: angles 0, 2pi/7, ...
    # Inner heptagon: angles pi/7, 3pi/7, ... (interleaved)
    # Ratio ~ 0.5 to 0.7 is typical for optimal packing
    
    angles_outer_init = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner_init = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 7, endpoint=False)
    
    # params: [7 outer angles, 7 inner angles, ratio, rotation, mod_outer, mod_inner]
    p0 = np.concatenate([
        angles_outer_init, 
        angles_inner_init, 
        [0.6, 0.0, 0.0, 0.0]
    ])
    
    best_score_val = 0.0
    best_result = None
    
    # Evaluate initial
    pts_init = _params_to_points(p0)
    best_score_val = _score(pts_init)
    best_result = pts_init.flatten()
    
    # Optimize structural parameters
    if time.time() < deadline - 60:
        try:
            res_struct = minimize(_objective_structural, p0, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            if res_struct.fun < 0:
                score_val = -res_struct.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_params = res_struct.x
                    best_result = _params_to_points(best_params).flatten()
        except Exception:
            pass

    # --- Phase 2: Cartesian Refinement ---
    # Now that we have a good topological configuration, refine in Cartesian space
    # This allows breaking the "perfect heptagon" assumption if beneficial
    
    if time.time() < deadline - 30:
        s, x = _powell(best_result, maxiter=2000)
        if s > best_score_val:
            best_score_val = s
            best_result = x

    # --- Phase 3: Perturbed Restarts in Cartesian Space ---
    # Try a few perturbations to escape local minima
    np.random.seed(42)
    for pert_scale in [0.01, 0.02]:
        if time.time() > deadline - 20:
            break
        x0_pert = best_result + np.random.normal(0, pert_scale, best_result.shape)
        s, x = _powell(x0_pert, maxiter=1000)
        if s > best_score_val:
            best_score_val = s
            best_result = x

    # --- Phase 4: Coordinate Descent ---
    if time.time() < deadline - 5:
        pts_best = best_result.reshape(N, 2)
        s_cd, pts_cd = _coordinate_descent(pts_best, n_iters=30, step=0.002)
        if s_cd > best_score_val:
            best_score_val = s_cd
            best_result = pts_cd.flatten()
    
    # --- Phase 5: Final Tight Polish ---
    if time.time() < deadline - 3 and best_score_val > 0:
        s_final, x_final = _powell(best_result, maxiter=500)
        if s_final > best_score_val:
            best_score_val = s_final
            best_result = x_final

    return best_result.reshape(N, 2)
