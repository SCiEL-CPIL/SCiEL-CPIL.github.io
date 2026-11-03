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
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf
    # Keep a list of top candidates for final polishing
    top_candidates = [] # List of (score, points)

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # Add to top candidates if it's good enough
        if score > 0:
            top_candidates.append((score, pts.copy()))
            # Keep only top 5 to limit memory and time in final phase
            if len(top_candidates) > 5:
                top_candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = top_candidates[:5]

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

    # --- Phase 1: Multi-Start Local Refinement ---
    # Generate diverse initial configurations in the symmetric space
    initial_configs = []
    
    # 1. Regular 14-gon (symmetric subset)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_configs.append(pts_reg[:N_HALF].flatten())
    
    # 2. Jittered Regular 14-gon
    for jitter in [0.1, 0.2, 0.3]:
        rng = np.random.default_rng(42 + int(jitter*100))
        pts_jit = pts_reg.copy()
        pts_jit += rng.normal(0, jitter, pts_jit.shape)
        initial_configs.append(pts_jit[:N_HALF].flatten())
        
    # 3. Grid-like structures
    # Try to form a 7x2 grid deformed into a circle
    for scale in [0.5, 0.8, 1.0]:
        x = np.linspace(-1, 1, 7) * scale
        y = np.linspace(-1, 1, 2) * scale
        grid = np.array([(xi, yi) for xi in x for yi in y])
        # Rotate and scale to fit in unit circle roughly
        theta = np.pi / 7
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        grid = grid @ rot.T
        norm = np.max(np.linalg.norm(grid, axis=1))
        if norm > 0:
            grid = grid / norm
        initial_configs.append(grid[:N_HALF].flatten())
        
    # 4. Random symmetric points
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.uniform(-1, 1, (N_HALF, 2))
        # Ensure no points are too close to origin or each other initially?
        # Just let the optimizer handle it.
        initial_configs.append(pts_rand.flatten())

    # Run local refinement on each initial config
    for i, init_flat in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break
            
        # Use soft-min L-BFGS-B for fast, robust convergence
        # Beta 50 is a good balance
        cand_pts, cand_score = refine_soft_sym(init_flat, method='L-BFGS-B', maxiter=2000, beta=50.0)
        if cand_pts is not None and cand_score > 0:
            update_best(cand_pts, cand_score)
            
            # If we found something good, do a quick hard refinement
            if cand_score > best_score * 0.95:
                 cand_pts2, cand_score2 = refine_sym(init_flat, method='Nelder-Mead', maxiter=1000, tol=1e-8)
                 if cand_pts2 is not None and cand_score2 > 0:
                     update_best(cand_pts2, cand_score2)

    # --- Phase 2: Refine Top Candidates with Higher Precision ---
    # Sort top candidates
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for score, pts in top_candidates[:3]: # Top 3
        if time.time() > deadline - 10.0:
            break
            
        half = pts[:N_HALF].flatten()
        
        # Try Powell in symmetric space
        cand_pts, cand_score = refine_sym(half, 'Powell', maxiter=2000, tol=1e-10)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        # Try Nelder-Mead with tighter tol
        cand_pts, cand_score = refine_sym(half, 'Nelder-Mead', maxiter=3000, tol=1e-10)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        # Try Soft-Min with higher Beta for sharper min
        cand_pts, cand_score = refine_soft_sym(half, 'L-BFGS-B', maxiter=3000, beta=100.0)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 3: Full-space refinement (break symmetry if beneficial) ---
    # Only try this if we have time, as it's 28D
    if best_points is not None and time.time() < deadline - 5.0:
        flat_full = best_points.flatten()
        cand_pts, cand_score = refine_full(flat_full, 'Nelder-Mead', maxiter=1000, tol=1e-9)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        cand_pts, cand_score = refine_soft_full(flat_full, 'L-BFGS-B', maxiter=2000, beta=50.0)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # --- Phase 4: Final high-precision polish ---
    if best_points is not None and time.time() < deadline - 1.0:
        half = best_points[:N_HALF].flatten()
        cand_pts, cand_score = refine_sym(half, 'Nelder-Mead', maxiter=5000, tol=1e-12)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
