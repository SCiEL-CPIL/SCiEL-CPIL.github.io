"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 145.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # 13*12*11/6 = 286 triangles. Loop is fast enough.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    if min_ta <= 0:
        return 0.0
    return min_ta / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _construct_symmetric_12_1(r_outer, r_inner, a_outer, a_inner):
    """
    12 points on outer circle, 1 point on inner circle (or center).
    Parameters:
        r_outer: radius of outer 12-gon
        r_inner: radius of inner point (distance from center)
        a_outer: rotation of outer 12-gon
        a_inner: angle of inner point
    """
    pts_outer = _regular_polygon(12, radius=r_outer, angle_offset=a_outer)
    # Inner point
    x_in = r_inner * np.cos(a_inner)
    y_in = r_inner * np.sin(a_inner)
    pts_inner = np.array([[x_in, y_in]])
    return np.vstack([pts_outer, pts_inner])


def _construct_symmetric_7_6(r_outer, r_inner, a_outer, a_inner):
    """
    7 points on outer circle, 6 points on inner circle.
    Parameters:
        r_outer: radius of outer 7-gon
        r_inner: radius of inner 6-gon
        a_outer: rotation of outer 7-gon
        a_inner: rotation of inner 6-gon
    """
    pts_outer = _regular_polygon(7, radius=r_outer, angle_offset=a_outer)
    pts_inner = _regular_polygon(6, radius=r_inner, angle_offset=a_inner)
    return np.vstack([pts_outer, pts_inner])


def _construct_symmetric_6_6_1(r_outer, r_mid, r_inner, a_outer, a_mid, a_inner):
    """
    6 points on outer, 6 on middle, 1 on inner (center).
    """
    pts_outer = _regular_polygon(6, radius=r_outer, angle_offset=a_outer)
    pts_mid = _regular_polygon(6, radius=r_mid, angle_offset=a_mid)
    x_in = r_inner * np.cos(a_inner)
    y_in = r_inner * np.sin(a_inner)
    pts_inner = np.array([[x_in, y_in]])
    return np.vstack([pts_outer, pts_mid, pts_inner])


def _optimize_symmetric_constructor(constructor_func, initial_params, param_bounds, max_iter=200):
    """
    Optimize parameters of a symmetric constructor using Nelder-Mead.
    """
    def objective(params):
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            return 0.0
        # Clip parameters to bounds to ensure validity during optimization steps
        params_clipped = np.clip(params, param_bounds[0], param_bounds[1])
        try:
            pts = constructor_func(*params_clipped)
            return -_score(pts)
        except Exception:
            return 0.0

    try:
        res = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < 0:
            params_opt = np.clip(res.x, param_bounds[0], param_bounds[1])
            pts = constructor_func(*params_opt)
            return pts, -res.fun
    except Exception:
        pass
    return None, 0.0


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # 1. Symmetric Search: 12-1 Split
    # This is the most likely candidate for optimal structure (one point in center, 12 on hull)
    # Parameters: r_outer, r_inner, a_outer, a_inner
    # Bounds: r_outer in [0.5, 2.0], r_inner in [0.0, 1.0], angles in [0, 2pi]
    
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 5:
        # Coarse grid search for 12-1
        best_121_pts = None
        best_121_score = 0.0
        
        # Grid over r_outer and r_inner. Angles are less critical due to symmetry but we optimize them.
        # Since the 12-gon has 12-fold symmetry, a_outer only matters relative to a_inner.
        # We can fix a_outer=0 and optimize a_inner relative to it, or just optimize both.
        
        r_outer_vals = np.linspace(0.8, 1.5, 15)
        r_inner_vals = np.linspace(0.0, 0.8, 10)
        
        for ro in r_outer_vals:
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            for ri in r_inner_vals:
                if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                    break
                # Try a few angular offsets
                for a_in in [0.0, np.pi/6, np.pi/3, np.pi/2]:
                    # Initial params: ro, ri, 0.0, a_in
                    init_params = [ro, ri, 0.0, a_in]
                    bounds = ([0.5, 0.0, -np.pi, -np.pi], [2.0, 1.5, np.pi, np.pi])
                    
                    pts, score = _optimize_symmetric_constructor(
                        _construct_symmetric_12_1, init_params, bounds, max_iter=100
                    )
                    if pts is not None and score > best_121_score:
                        best_121_score = score
                        best_121_pts = pts
                        
        if best_121_pts is not None and best_121_score > best_score:
            best_score = best_121_score
            best_pts = best_121_pts

    # 2. Symmetric Search: 7-6 Split
    # More balanced, might be better if 12-1 creates small triangles involving the center point and adjacent hull points.
    
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 5:
        best_76_pts = None
        best_76_score = 0.0
        
        r_outer_vals = np.linspace(0.8, 1.5, 15)
        r_inner_vals = np.linspace(0.2, 0.9, 15)
        
        for ro in r_outer_vals:
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            for ri in r_inner_vals:
                if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                    break
                # Relative angle is key
                for a_rel in [0.0, np.pi/6, np.pi/12]:
                    init_params = [ro, ri, 0.0, a_rel]
                    bounds = ([0.5, 0.1, -np.pi, -np.pi], [2.0, 1.5, np.pi, np.pi])
                    
                    pts, score = _optimize_symmetric_constructor(
                        _construct_symmetric_7_6, init_params, bounds, max_iter=100
                    )
                    if pts is not None and score > best_76_score:
                        best_76_score = score
                        best_76_pts = pts
                        
        if best_76_pts is not None and best_76_score > best_score:
            best_score = best_76_score
            best_pts = best_76_pts

    # 3. Symmetric Search: 6-6-1 Split
    # Three rings
    
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 5:
        best_661_pts = None
        best_661_score = 0.0
        
        r_outer_vals = np.linspace(0.9, 1.4, 10)
        r_mid_vals = np.linspace(0.4, 0.8, 10)
        
        for ro in r_outer_vals:
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            for rm in r_mid_vals:
                if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                    break
                    # Inner is usually 0 or small
                    for ri in [0.0, 0.2]:
                        init_params = [ro, rm, ri, 0.0, 0.0, 0.0]
                        bounds = ([0.5, 0.2, 0.0, -np.pi, -np.pi, -np.pi], 
                                  [2.0, 1.2, 0.5, np.pi, np.pi, np.pi])
                        
                        pts, score = _optimize_symmetric_constructor(
                            _construct_symmetric_6_6_1, init_params, bounds, max_iter=100
                        )
                        if pts is not None and score > best_661_score:
                            best_661_score = score
                            best_661_pts = pts

        if best_661_pts is not None and best_661_score > best_score:
            best_score = best_661_score
            best_pts = best_661_pts

    # 4. Unconstrained Refinement
    # Take the best symmetric solution and allow it to break symmetry slightly to find local maxima in the full 26D space.
    
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 10:
        x0 = best_pts.flatten()
        
        def full_objective(x):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                return 0.0
            pts = x.reshape(-1, 2)
            return -_score(pts)
            
        try:
            res = minimize(
                full_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < -best_score:
                candidate = res.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass

    # 5. Final Polish with tighter tolerances if time remains
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 5:
        x0 = best_pts.flatten()
        
        def full_objective(x):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                return 0.0
            pts = x.reshape(-1, 2)
            return -_score(pts)
            
        try:
            res = minimize(
                full_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < -best_score:
                candidate = res.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass

    # Fallback check
    final_score = _score(best_pts)
    if final_score <= 0:
        best_pts = _regular_polygon(N)

    return best_pts
# EVOLVE-BLOCK-END
