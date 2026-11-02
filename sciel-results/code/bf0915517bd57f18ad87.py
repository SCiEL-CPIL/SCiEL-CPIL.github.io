"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
N_HALF = 7

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective_soft_sym(flat, beta=50.0):
    """Soft-min objective for centrally symmetric parameterization."""
    pts_half = flat.reshape(N_HALF, 2)
    pts = np.vstack([pts_half, -pts_half])
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_sym(flat):
    """Hard objective for centrally symmetric parameterization."""
    pts_half = flat.reshape(N_HALF, 2)
    pts = np.vstack([pts_half, -pts_half])
    return -_score(pts)

def _objective_soft_full(flat, beta=50.0):
    """Soft-min objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_full(flat):
    """Hard objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    return -_score(pts)

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _normalize(pts):
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        return pts

    def refine_sym(flat_init, method='Nelder-Mead', maxiter=2000, tol=1e-10):
        """Refine in the 14D symmetric space."""
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective_sym, flat_init, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(_objective_sym, flat_init, method='Powell',
                               options={'maxiter': maxiter, 'xatol': tol, 'ftol': tol})
            elif method == 'L-BFGS-B':
                res = minimize(_objective_sym, flat_init, method='L-BFGS-B',
                               options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            else:
                return None, -np.inf
            cand_pts = np.vstack([res.x.reshape(N_HALF, 2), -res.x.reshape(N_HALF, 2)])
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft_sym(flat_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min in symmetric space."""
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft_sym(x, beta=beta), flat_init,
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft_sym(x, beta=beta), flat_init,
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft_sym(x, beta=beta), flat_init,
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            cand_pts = np.vstack([res.x.reshape(N_HALF, 2), -res.x.reshape(N_HALF, 2)])
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_full(flat_init, method='Nelder-Mead', maxiter=2000, tol=1e-10):
        """Refine in the full 28D space."""
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective_full, flat_init, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(_objective_full, flat_init, method='Powell',
                               options={'maxiter': maxiter, 'xatol': tol, 'ftol': tol})
            elif method == 'L-BFGS-B':
                res = minimize(_objective_full, flat_init, method='L-BFGS-B',
                               options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            else:
                return None, -np.inf
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft_full(flat_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min in full space."""
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), flat_init,
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), flat_init,
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), flat_init,
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # --- Phase 1: Symmetric Differential Evolution (14D) ---
    if time.time() < deadline - 50.0:
        def de_sym_obj(flat):
            pts_half = flat.reshape(N_HALF, 2)
            pts = np.vstack([pts_half, -pts_half])
            if np.allclose(pts, pts[0]):
                return 1e10
            score = _score(pts)
            return -score

        try:
            from scipy.optimize import differential_evolution
            res_de = differential_evolution(
                de_sym_obj,
                bounds=[(-1, 1)] * (N_HALF * 2),
                maxiter=100,
                tol=1e-6,
                seed=42,
                polish=False,
                popsize=15
            )
            if res_de.success or res_de.fun < 0:
                cand_pts = np.vstack([res_de.x.reshape(N_HALF, 2), -res_de.x.reshape(N_HALF, 2)])
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # If DE didn't find anything good, fall back to regular polygon
    if best_points is None or best_score < 0.1:
        angles = np.linspace(0, 2 * np.pi, N_HALF, endpoint=False)
        pts_half = np.column_stack([np.cos(angles), np.sin(angles)])
        init_pts = np.vstack([pts_half, -pts_half])
        update_best(init_pts, _score(init_pts))

    # --- Phase 2: Refine with L-BFGS-B on soft objective (Symmetric) ---
    if time.time() < deadline - 30.0:
        flat_init = best_points[:N_HALF].flatten()
        cand_pts, cand_score = refine_soft_sym(flat_init, method='L-BFGS-B', maxiter=2000, beta=50.0)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        # Try higher beta for sharper min
        if time.time() < deadline - 20.0:
            cand_pts, cand_score = refine_soft_sym(flat_init, method='L-BFGS-B', maxiter=2000, beta=100.0)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # --- Phase 3: Refine with Nelder-Mead on hard objective (Symmetric) ---
    if time.time() < deadline - 20.0:
        flat_init = best_points[:N_HALF].flatten()
        cand_pts, cand_score = refine_sym(flat_init, method='Nelder-Mead', maxiter=1000, tol=1e-12)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 4: Break Symmetry? Try Full Space Refinement ---
    # The inspiration suggests full space might be better, but symmetric is often optimal for even N.
    # Let's try a full space refinement from the best symmetric point.
    if time.time() < deadline - 15.0:
        flat_init = best_points.flatten()
        # Use soft objective for full space to navigate
        cand_pts, cand_score = refine_soft_full(flat_init, method='L-BFGS-B', maxiter=1000, beta=50.0)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        # Polish with hard objective
        if time.time() < deadline - 5.0:
            flat_init = best_points.flatten()
            cand_pts, cand_score = refine_full(flat_init, method='Nelder-Mead', maxiter=500, tol=1e-12)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # --- Final Normalization ---
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        final_points = best_points

    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        
    return final_points
# EVOLVE-BLOCK-END
