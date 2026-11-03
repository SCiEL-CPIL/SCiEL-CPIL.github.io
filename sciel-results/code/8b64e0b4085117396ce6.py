"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

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

# --- Support function parameterization ---
# A convex body is defined by its support function h(θ) = max_i x_i·(cosθ, sinθ)
# For a polygon with vertices in CCW order, h(θ) is piecewise linear in θ.
# We parameterize by the angles of the normal directions and the support values.

def _support_function_to_points(norm_angles, support_vals, n_samples=360):
    """
    Convert support function parameters to polygon vertices.
    norm_angles: sorted angles of normal directions (len = M)
    support_vals: support function values at those angles (len = M)
    Returns polygon vertices in CCW order.
    """
    M = len(norm_angles)
    if M < 3:
        return np.zeros((3, 2))
    
    # Ensure sorted
    order = np.argsort(norm_angles)
    norm_angles = norm_angles[order]
    support_vals = support_vals[order]
    
    # Build the support function on a fine grid
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    # Interpolate support function (circular)
    # Extend by one period for circular interpolation
    na_ext = np.concatenate([norm_angles - 2*np.pi, norm_angles, norm_angles + 2*np.pi])
    sv_ext = np.concatenate([support_vals, support_vals, support_vals])
    
    h_thetas = np.interp(thetas, na_ext, sv_ext, period=2*np.pi)
    
    # The boundary point in direction θ is where the support line touches.
    # For a smooth support function, the point is (h(θ)cosθ + h'(θ)sinθ, h(θ)sinθ - h'(θ)cosθ)
    # But for a polygon, we find the vertices as intersections of adjacent support lines.
    
    # Find vertices by intersecting adjacent support lines
    vertices = []
    for i in range(M):
        i1 = i
        i2 = (i + 1) % M
        # Support lines: x·n_i = h_i and x·n_{i+1} = h_{i+1}
        n1 = np.array([np.cos(norm_angles[i1]), np.sin(norm_angles[i1])])
        n2 = np.array([np.cos(norm_angles[i2]), np.sin(norm_angles[i2])])
        h1 = support_vals[i1]
        h2 = support_vals[i2]
        
        # Solve [n1; n2] · x = [h1; h2]
        A = np.array([n1, n2])
        b = np.array([h1, h2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            continue
        x = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
        y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
        vertices.append(np.array([x, y]))
    
    if len(vertices) < 3:
        return np.zeros((3, 2))
    
    verts = np.array(vertices)
    # Sort by angle around centroid to ensure CCW order
    centroid = np.mean(verts, axis=0)
    angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return verts[order]

def _support_score(params, n_samples=360):
    """Compute score from support function parameters."""
    M = N  # Use N vertices
    norm_angles = params[:M]
    support_vals = params[M:]
    
    # Ensure angles are sorted and in [0, 2π)
    norm_angles = np.sort(norm_angles) % (2 * np.pi)
    support_vals = np.sort(support_vals)
    
    # Ensure positive support values
    support_vals = np.maximum(support_vals, 0.1)
    
    pts = _support_function_to_points(norm_angles, support_vals, n_samples)
    if len(pts) < 3:
        return 0.0
    return _score(pts)

def _support_objective(params, n_samples=360):
    """Negative score for support function optimization."""
    return -_support_score(params, n_samples)

def _support_soft_objective(params, beta=50.0, n_samples=360):
    """Soft-min objective for support function parameterization."""
    M = N
    norm_angles = params[:M]
    support_vals = params[M:]
    
    norm_angles = np.sort(norm_angles) % (2 * np.pi)
    support_vals = np.sort(support_vals)
    support_vals = np.maximum(support_vals, 0.1)
    
    pts = _support_function_to_points(norm_angles, support_vals, n_samples)
    if len(pts) < 3 or np.any(~np.isfinite(pts)):
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

# --- Symmetric parameterization (7 half-points + 7 negatives) ---

def _params_to_points(params):
    """
    Convert 14 parameters (7 angles + 7 radii) to 14 centrally symmetric points.
    """
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective_sym(params):
    """Negative score for symmetric parameterization."""
    points = _params_to_points(params)
    return -_score(points)

def _objective_soft_sym(params, beta=50.0):
    """Soft-min objective for centrally symmetric parameterization."""
    pts_half = _params_to_points(params)[:N_HALF]
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

# --- Full 28D parameterization ---

def _objective_full(flat):
    """Negative score for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
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

def construct():
    """
    Construct 14 points maximizing min triangle area / convex hull area.
    Uses support function parameterization as primary approach with symmetric
    fallback and multi-start local refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_points = None
    best_score = -np.inf
    rng = np.random.default_rng(42)

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # --- Phase 1: Support function optimization (28D: 14 angles + 14 support values) ---
    # Initialize with regular polygon support function
    M = N
    base_angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    base_support = np.ones(M)
    
    support_inits = []
    # Regular polygon
    support_inits.append(np.concatenate([base_angles, base_support]))
    
    # Perturbed regular
    for scale in [0.8, 1.0, 1.2]:
        support_inits.append(np.concatenate([base_angles, base_support * scale]))
    
    # Random perturbations of angles and support
    for _ in range(15):
        ang_pert = base_angles + rng.normal(0, 0.1, M)
        ang_pert = np.sort(ang_pert % (2 * np.pi))
        sup_pert = base_support + rng.normal(0, 0.15, M)
        sup_pert = np.maximum(sup_pert, 0.3)
        support_inits.append(np.concatenate([ang_pert, sup_pert]))
    
    # More diverse random inits
    for _ in range(10):
        ang_rand = np.sort(rng.uniform(0, 2 * np.pi, M))
        sup_rand = rng.uniform(0.5, 1.5, M)
        support_inits.append(np.concatenate([ang_rand, sup_rand]))
    
    # Score and sort initial candidates
    scored_support = []
    for init_params in support_inits:
        if time.time() > deadline - 60:
            break
        try:
            s = _support_score(init_params)
            scored_support.append((s, init_params))
        except Exception:
            pass
    scored_support.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates
    for s, p in scored_support[:8]:
        if time.time() > deadline - 30:
            break
        
        # L-BFGS-B with soft objective
        try:
            res_soft = minimize(
                lambda pp: _support_soft_objective(pp, beta=60.0),
                p,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if np.isfinite(res_soft.fun):
                M_curr = N
                na = np.sort(res_soft.x[:M_curr]) % (2 * np.pi)
                sv = np.sort(np.maximum(res_soft.x[M_curr:], 0.1))
                pts = _support_function_to_points(na, sv)
                if len(pts) >= 3:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_points = pts.copy()
        except Exception:
            pass
        
        # Nelder-Mead on hard objective
        try:
            res_nm = minimize(
                _support_objective,
                p,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.isfinite(res_nm.fun):
                M_curr = N
                na = np.sort(res_nm.x[:M_curr]) % (2 * np.pi)
                sv = np.sort(np.maximum(res_nm.x[M_curr:], 0.1))
                pts = _support_function_to_points(na, sv)
                if len(pts) >= 3:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_points = pts.copy()
        except Exception:
            pass

    # --- Phase 2: Symmetric parameterization (14D) as fallback/complement ---
    inits_sym = []
    base_angles_sym = np.linspace(0.1, np.pi - 0.1, N_HALF)
    inits_sym.append(np.concatenate([base_angles_sym, np.ones(N_HALF)]))
    for r_scale in [0.7, 1.0, 1.3]:
        inits_sym.append(np.concatenate([base_angles_sym, np.full(N_HALF, r_scale)]))
    for _ in range(10):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        radii = rng.uniform(0.5, 1.5, N_HALF)
        inits_sym.append(np.concatenate([angles, radii]))
    
    scored_sym = []
    for init_params in inits_sym:
        if time.time() > deadline - 30:
            break
        try:
            s = -_objective_sym(init_params)
            scored_sym.append((s, init_params))
        except Exception:
            pass
    scored_sym.sort(key=lambda x: x[0], reverse=True)
    
    for s, p in scored_sym[:5]:
        if time.time() > deadline - 15:
            break
        
        try:
            res_soft = minimize(
                lambda pp: _objective_soft_sym(pp, beta=60.0),
                p,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if np.isfinite(res_soft.fun):
                pts = _params_to_points(res_soft.x)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
        except Exception:
            pass
        
        try:
            res_nm = minimize(
                _objective_sym,
                p,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.isfinite(res_nm.fun):
                pts = _params_to_points(res_nm.x)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
        except Exception:
            pass

    # --- Phase 3: Full 28D refinement from best solution ---
    if best_points is not None and time.time() < deadline - 15:
        flat_init = best_points.flatten()
        
        try:
            res_soft_full = minimize(
                lambda x: _objective_soft_full(x, beta=50.0),
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if np.isfinite(res_soft_full.fun):
                pts = res_soft_full.x.reshape(N, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
        except Exception:
            pass
        
        try:
            res_nm_full = minimize(
                _objective_full,
                flat_init,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.isfinite(res_nm_full.fun):
                pts = res_nm_full.x.reshape(N, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
        except Exception:
            pass

    # --- Phase 4: Coordinate-wise fine search ---
    if best_points is not None and time.time() < deadline - 5:
        for coord_iter in range(60):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                for dim in range(2):
                    for delta in [-0.005, -0.002, 0.002, 0.005]:
                        candidate = best_points.copy()
                        candidate[i, dim] += delta
                        sc = _score(candidate)
                        if sc > best_score + 1e-12:
                            best_score = sc
                            best_points = candidate.copy()
                            improved = True
            if not improved:
                break

    # --- Phase 5: Final Nelder-Mead polish ---
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if np.isfinite(res_final.fun):
                pts = res_final.x.reshape(N, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
        except Exception:
            pass

    if best_points is None:
        theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])

    return best_points
