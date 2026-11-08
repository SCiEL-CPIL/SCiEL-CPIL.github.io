import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    # Base angles for the 13 points
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Fourier modes: K=4
    # We remove k=1 terms (a1, b1) to avoid redundancy with cx, cy.
    # The shape is defined by r0 and harmonics k=2..K.
    # Parameters: cx, cy, r0, a2, b2, a3, b3, a4, b4
    # Total 2 + 1 + 2*(K-1) = 9 parameters?
    # K=4 means k=2,3,4. So 3 harmonics. 2*3=6 params.
    # Total 9 params.
    
    K = 4
    n_harmonics = K - 1  # k=2 to K
    n_params = 2 + 1 + 2 * n_harmonics  # cx, cy, r0, a2..bK
    
    def params_to_points(params):
        cx, cy, r0 = params[0], params[1], params[2]
        # a, b start from index 3
        # a[0] corresponds to k=2, a[1] to k=3, etc.
        a = params[3:3+n_harmonics]
        b = params[3+n_harmonics:3+2*n_harmonics]
        
        # Compute r(theta)
        r = np.full(N, r0)
        for idx, k in enumerate(range(2, K+1)):
            r += a[idx] * np.cos(k * base_angles)
            r += b[idx] * np.sin(k * base_angles)
        
        # Enforce positive radius
        r = np.maximum(r, 0.1)
        
        x = cx + r * np.cos(base_angles)
        y = cy + r * np.sin(base_angles)
        return np.column_stack([x, y])

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def soft_objective(params, beta=100.0):
        """
        Objective: minimize -soft_min(triangle areas) for unit hull area.
        """
        pts = params_to_points(params)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def hard_objective(params):
        """
        Objective: minimize -min(triangle areas) for unit hull area.
        """
        pts = params_to_points(params)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Differential Evolution bounds
    # cx, cy in [-0.5, 0.5]
    # r0 in [0.5, 2.0]
    # a_k, b_k in [-0.5, 0.5]
    bounds = [
        (-0.5, 0.5),  # cx
        (-0.5, 0.5),  # cy
        (0.5, 2.0),   # r0
    ]
    for _ in range(n_harmonics):
        bounds.append((-0.5, 0.5))  # a_k
        bounds.append((-0.5, 0.5))  # b_k

    best_score = -float('inf')
    best_params = None

    # 1. Differential Evolution
    # Reduced dimensionality (9 params instead of 11) should allow for larger popsize/iterations
    try:
        de_res = opt.differential_evolution(
            soft_objective,
            bounds,
            args=(100.0,),
            maxiter=100,
            popsize=30,
            tol=1e-5,
            seed=42,
            polish=False,
            init='sobol'
        )
        best_params = de_res.x
        best_score = -soft_objective(best_params, 100.0)
    except Exception as e:
        # Fallback to regular polygon
        best_params = np.zeros(n_params)
        best_params[2] = 1.0
        best_score = compute_score(params_to_points(best_params))

    # 2. Refinement using Powell on hard objective
    # Powell is robust for non-smooth functions
    if time.time() < start_time + time_limit - 10:
        try:
            res = opt.minimize(
                hard_objective,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate_params = res.x
            candidate_score = -hard_objective(candidate_params)
            if candidate_score > best_score:
                best_params = candidate_params
                best_score = candidate_score
        except Exception:
            pass

    # 3. Try Nelder-Mead as an alternative local optimizer
    if time.time() < start_time + time_limit - 5:
        try:
            res = opt.minimize(
                hard_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            candidate_params = res.x
            candidate_score = -hard_objective(candidate_params)
            if candidate_score > best_score:
                best_params = candidate_params
                best_score = candidate_score
        except Exception:
            pass

    # Convert best params to points
    best_points = params_to_points(best_params)
    
    # Final check against regular polygon
    reg_poly_params = np.zeros(n_params)
    reg_poly_params[2] = 1.0
    reg_poly_points = params_to_points(reg_poly_params)
    reg_poly_score = compute_score(reg_poly_points)
    
    if reg_poly_score > best_score:
        best_points = reg_poly_points
        best_score = reg_poly_score

    return best_points
