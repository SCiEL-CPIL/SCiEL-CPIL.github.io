import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

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
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    if time.time() < deadline - 60.0:
        def de_sym_obj(flat):
            pts_half = flat.reshape(N_HALF, 2)
            pts = np.vstack([pts_half, -pts_half])
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)

        try:
            bounds = [(-1, 1)] * 14
            res_de = differential_evolution(
                de_sym_obj, bounds, seed=42, maxiter=80, popsize=15,
                tol=1e-6, mutation=(0.5, 1.0), recombination=0.9,
                polish=True, init='latinhypercube'
            )
            if res_de.fun < 0:
                pts7 = res_de.x.reshape(N_HALF, 2)
                pts14 = np.vstack([pts7, -pts7])
                update_best(pts14, -res_de.fun)
        except Exception:
            pass

    # --- Phase 2: Symmetric DE with different seeds ---
    if time.time() < deadline - 45.0:
        for seed in [7, 99, 1234]:
            if time.time() > deadline - 45.0:
                break
            def de_sym_obj2(flat):
                pts_half = flat.reshape(N_HALF, 2)
                pts = np.vstack([pts_half, -pts_half])
                if np.allclose(pts, pts[0]):
                    return 1.0
                return -_score(pts)
            try:
                bounds = [(-1, 1)] * 14
                res_de = differential_evolution(
                    de_sym_obj2, bounds, seed=seed, maxiter=50, popsize=12,
                    tol=1e-5, mutation=(0.3, 0.9), recombination=0.8,
                    polish=True, init='latinhypercube'
                )
                if res_de.fun < 0:
                    pts7 = res_de.x.reshape(N_HALF, 2)
                    pts14 = np.vstack([pts7, -pts7])
                    update_best(pts14, -res_de.fun)
            except Exception:
                pass

    # --- Phase 3: Nelder-Mead refinement of DE results in symmetric space ---
    if best_points is not None and time.time() < deadline - 35.0:
        # Extract symmetric half from best
        # Try to find the best symmetric representation
        # Just use the first 7 points as the half
        half = best_points[:N_HALF].flatten()
        cand_pts, cand_score = refine_sym(half, 'Nelder-Mead', maxiter=3000, tol=1e-10)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 4: Powell refinement in symmetric space ---
    if best_points is not None and time.time() < deadline - 25.0:
        half = best_points[:N_HALF].flatten()
        cand_pts, cand_score = refine_sym(half, 'Powell', maxiter=2000, tol=1e-10)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 5: Soft-min refinement in symmetric space ---
    if best_points is not None and time.time() < deadline - 15.0:
        half = best_points[:N_HALF].flatten()
        for beta in [30.0, 50.0, 100.0]:
            if time.time() > deadline - 15.0:
                break
            cand_pts, cand_score = refine_soft_sym(half, 'L-BFGS-B', maxiter=3000, beta=beta)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # --- Phase 6: Full-space refinement (break symmetry if beneficial) ---
    if best_points is not None and time.time() < deadline - 10.0:
        flat_full = best_points.flatten()
        cand_pts, cand_score = refine_full(flat_full, 'Nelder-Mead', maxiter=2000, tol=1e-10)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 7: Soft-min full-space refinement ---
    if best_points is not None and time.time() < deadline - 5.0:
        flat_full = best_points.flatten()
        cand_pts, cand_score = refine_soft_full(flat_full, 'L-BFGS-B', maxiter=3000, beta=50.0)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 8: Final high-precision polish ---
    if best_points is not None and time.time() < deadline - 2.0:
        half = best_points[:N_HALF].flatten()
        cand_pts, cand_score = refine_sym(half, 'Nelder-Mead', maxiter=5000, tol=1e-12)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
