import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area(points):
    # Vectorized calculation of min triangle area
    n = len(points)
    min_area = float('inf')
    # 13*12*11/6 = 286 triangles. Loop is fast enough and memory safe.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, center=(0.0, 0.0), radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _construct_symmetric(r_outer, r_inner, n_outer, n_inner, angle_offset_outer=0.0, angle_offset_inner=0.0):
    """
    Construct points using two concentric regular polygons.
    n_outer + n_inner must equal N.
    """
    pts_outer = _regular_polygon(n_outer, radius=r_outer, angle_offset=angle_offset_outer)
    pts_inner = _regular_polygon(n_inner, radius=r_inner, angle_offset=angle_offset_inner)
    return np.vstack([pts_outer, pts_inner])

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin (limit is 120s or 150s, use 110 to be safe)
    
    # 1. Regular Polygon Baseline (all 13 on circle)
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # 2. Symmetric Parameter Search
    # We parameterize the points as two concentric regular polygons.
    # Possible splits for N=13: (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)
    # (7, 6) is the most balanced. (12, 1) is like a regular 13-gon with one point moved in.
    # Let's try a few splits and optimize r_outer, r_inner, and angles.
    
    def _symmetric_objective(params):
        if time.time() - start_time > deadline:
            return 0.0
        r_outer, r_inner, a_outer, a_inner = params
        # Ensure positive radii
        r_outer = abs(r_outer) + 0.1
        r_inner = abs(r_inner) + 0.01
        
        n_outer, n_inner = 7, 6
        pts = _construct_symmetric(r_outer, r_inner, n_outer, n_inner, a_outer, a_inner)
        return -_score(pts)
    
    # Try split 7-6
    if time.time() - start_time < deadline - 10:
        try:
            # Initial guess: r_outer=1, r_inner=0.5
            x0 = [1.0, 0.5, 0.0, 0.0]
            res = minimize(_symmetric_objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_score:
                r_o, r_i, a_o, a_i = res.x
                r_o = abs(r_o) + 0.1
                r_i = abs(r_i) + 0.01
                candidate = _construct_symmetric(r_o, r_i, 7, 6, a_o, a_i)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # Try split 12-1 (essentially optimizing the radius of the center point and rotation of the 12-gon)
    if time.time() - start_time < deadline - 10:
        def _symmetric_objective_121(params):
            if time.time() - start_time > deadline:
                return 0.0
            r_outer, r_inner, a_outer, a_inner = params
            r_outer = abs(r_outer) + 0.1
            r_inner = abs(r_inner) # Center point can be anywhere, but let's constrain to radius
            # For 1 inner point, angle doesn't matter for symmetry, but let's keep it for consistency
            # Actually, for 1 point, it's just at (r_inner*cos(a), r_inner*sin(a))
            pts_outer = _regular_polygon(12, radius=r_outer, angle_offset=a_outer)
            # Inner point
            x_in = r_inner * np.cos(a_inner)
            y_in = r_inner * np.sin(a_inner)
            pts_inner = np.array([[x_in, y_in]])
            return -_score(np.vstack([pts_outer, pts_inner]))
        
        try:
            x0 = [1.0, 0.0, 0.0, 0.0]
            res = minimize(_symmetric_objective_121, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_score:
                r_o, r_i, a_o, a_i = res.x
                r_o = abs(r_o) + 0.1
                r_i = abs(r_i)
                pts_outer = _regular_polygon(12, radius=r_o, angle_offset=a_o)
                x_in = r_i * np.cos(a_i)
                y_in = r_i * np.sin(a_i)
                pts_inner = np.array([[x_in, y_in]])
                candidate = np.vstack([pts_outer, pts_inner])
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 3. Differential Evolution on the full 26-dimensional space
    # This helps escape local minima of the symmetric constraints if they are suboptimal
    if time.time() - start_time < deadline - 30:
        bounds = [(-1.5, 1.5)] * (2 * N)
        
        def objective(x):
            if time.time() - start_time > deadline:
                return 0.0
            points = x.reshape(-1, 2)
            return -_score(points)
            
        try:
            # Use DE to find a good candidate
            # popsize=15, maxiter=50 is a reasonable start
            res_de = differential_evolution(objective, bounds, maxiter=30, popsize=15, tol=1e-4, seed=42, polish=False, init='latinhypercube')
            
            if res_de.fun < -best_score:
                candidate = res_de.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 4. Local Refinement from Best Points
    # Try Nelder-Mead from the current best
    if time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        try:
            res_nm = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_nm.fun < -best_score:
                candidate = res_nm.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 5. Try Nelder-Mead from Regular Polygon (in case DE drifted)
    if time.time() - start_time < deadline - 5:
        x0_reg = _regular_polygon(N).flatten()
        try:
            res_nm_reg = minimize(objective, x0_reg, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_nm_reg.fun < -best_score:
                candidate = res_nm_reg.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    return best_points
