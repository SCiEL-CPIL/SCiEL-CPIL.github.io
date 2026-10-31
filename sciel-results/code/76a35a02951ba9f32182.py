import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_objective(params, beta):
    """
    Soft max surrogate for gradient-based optimization in reduced space.
    P[0]=(0,0), P[1]=(1,0), so min distance is at most 1.
    We want to minimize the max squared distance.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max via log-sum-exp (numerically stable)
    max_d_sq = np.max(d_sq)
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    return soft_max

def _exact_objective(params):
    """Exact objective in reduced parameter space: max squared distance."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return np.max(d_sq) / min_d_sq

def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    This removes translation (2), rotation (1), and scale (1) = 4 DOF.
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    iu = np.triu_indices(N, 1)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[i_min]
    p1 = P[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform: translate so p0 is origin, rotate, scale so p1 is at (1,0)
    transformed = (P - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, rest in original order
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    params = transformed[2:].flatten()
    return params

def _reconstruct(params):
    """Reconstruct full point set from reduced parameters."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return P

def make_hexagonal_cluster(params):
    """
    Generate 16 points as a hexagonal lattice cluster with 3 parameters:
    params[0] = aspect ratio (y-scale relative to x)
    params[1] = shear angle (radians)
    params[2] = radial scaling factor for outer ring
    """
    aspect = params[0]
    shear = params[1]
    outer_scale = params[2]
    
    base_points = []
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0
        base_points.append([x, y])
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - np.sqrt(3)/2
        base_points.append([x, y])
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0 - np.sqrt(3)
        base_points.append([x, y])
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - 1.5*np.sqrt(3)
        base_points.append([x, y])
    for i in range(2):
        x = (i - 0.5) * 1.0
        y = 2.0 - 2.0*np.sqrt(3)
        base_points.append([x, y])
    
    base_points = np.array(base_points, dtype=float)
    
    centroid = base_points.mean(axis=0)
    base_points -= centroid
    
    base_points[:, 1] *= aspect
    
    shear_matrix = np.array([[1, np.tan(shear)], [0, 1]])
    base_points = base_points @ shear_matrix.T
    
    dists_from_center = np.linalg.norm(base_points, axis=1)
    median_dist = np.median(dists_from_center)
    outer_mask = dists_from_center > median_dist
    
    if np.any(outer_mask):
        outer_points = base_points[outer_mask]
        outer_dists = np.linalg.norm(outer_points, axis=1)
        outer_dists[outer_dists < 1e-12] = 1e-12
        scale_factors = outer_scale / outer_dists
        base_points[outer_mask] *= scale_factors[:, np.newaxis]
    
    return base_points

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # Hexagonal lattice subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    candidates.append(hex_points[sorted_indices][:16])
    
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # Concentric circles
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 1 center + 5 + 10
    pts = [np.array([0.0, 0.0])]
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    candidates.append(np.array(pts))
    
    # Golden angle spiral
    golden_angle = np.pi * (3. - np.sqrt(5.))
    angles = np.arange(16) * golden_angle
    radii = np.sqrt(np.arange(16)) / np.sqrt(16) * 2.0
    spiral = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(spiral)
    candidates.append(spiral + rng.normal(0, 0.1, (16, 2)))
    
    # Random uniform
    for _ in range(4):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # Hexagonal cluster with various parameters
    hex_params_list = [
        [1.0, 0.0, 1.0],
        [1.1, 0.1, 1.05],
        [0.9, -0.1, 0.95],
        [1.2, 0.2, 1.1],
        [0.8, 0.0, 1.2],
        [1.0, 0.0, 0.9],
    ]
    for params in hex_params_list:
        pts = make_hexagonal_cluster(np.array(params))
        candidates.append(pts)
    
    return candidates

def _local_optimize(params, maxiter_nm=200, maxiter_lbfgs=300):
    """
    Two-phase local optimization:
    1. Nelder-Mead on exact objective
    2. L-BFGS-B on soft objective with annealing
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Nelder-Mead
    try:
        res_nm = minimize(
            _exact_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x.copy()
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with annealing on soft objective
    betas = [10.0, 50.0, 200.0, 1000.0]
    for beta in betas:
        try:
            res_lbfgs = minimize(
                lambda p: _soft_objective(p, beta),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            exact_s = _exact_objective(res_lbfgs.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Local optimization from diverse starts
    for P0 in candidates:
        if time.time() > deadline - 25:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            p_opt, score = _local_optimize(params, maxiter_nm=200, maxiter_lbfgs=300)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Random restarts in reduced space
    while time.time() < deadline - 25:
        try:
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            p_opt, score = _local_optimize(params, maxiter_nm=150, maxiter_lbfgs=200)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 10:
        try:
            # Very high beta for near-exact optimization
            res_final = minimize(
                lambda p: _soft_objective(p, 5000.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
            )
            exact_s = _exact_objective(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass
        
        # Final Nelder-Mead polish
        if time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    _exact_objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return _reconstruct(best_params)
