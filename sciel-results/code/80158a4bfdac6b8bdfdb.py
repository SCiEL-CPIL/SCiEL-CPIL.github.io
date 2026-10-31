"""Optimized 16-point placement: minimize (max_dist/min_dist)^2."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2
DEADLINE = 140.0  # safety margin before 150s kill


def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]


def _exact_score(P):
    d2 = _pairwise_dists_sq(P)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)


def _soft_objective(params, beta=50.0):
    """
    Soft min/max surrogate for gradient-based optimization in reduced space.
    params: (28,) flattened 14 points (p0=(0,0), p1=(1,0) fixed).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d2 = _pairwise_dists_sq(P)
    d2 = np.maximum(d2, 1e-12)
    
    # Soft max via log-sum-exp
    max_d2 = np.max(d2)
    soft_max = max_d2 + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d2 - max_d2))))
    
    # Soft min
    min_d2 = np.min(d2)
    soft_min = min_d2 - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d2 - min_d2))))
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min


def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _exact_score(P)


def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    """
    d2 = _pairwise_dists_sq(P)
    min_idx = np.argmin(d2)
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
    """Reconstruct points from params (p0=(0,0), p1=(1,0), rest=params)."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return P


def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p ** 2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points ** 2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    candidates.append(hex_points[sorted_indices][:16])
    
    # Jittered hex
    for sigma in [0.05, 0.1, 0.2]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 3. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 4. Random uniform
    for _ in range(5):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 5. Random on ring
    for _ in range(3):
        angles = rng.uniform(0, 2 * np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    return candidates


def _local_optimize_params(params, maxiter_nm=500, maxiter_lbfgs=500):
    """
    Two-phase local optimization in reduced parameter space:
    1. Nelder-Mead on exact objective
    2. L-BFGS-B on soft objective for gradient-based polish
    Returns best params and exact score.
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
    
    # Phase 2: L-BFGS-B on soft objective
    try:
        res_lbfgs = minimize(
            lambda p: _soft_objective(p, beta=50.0),
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
    t0 = time.time()
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Local optimization from diverse starts
    for P0 in candidates:
        if time.time() > DEADLINE - 20:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            p_opt, score = _local_optimize_params(params, maxiter_nm=300, maxiter_lbfgs=300)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Random restarts in reduced space
    while time.time() < DEADLINE - 20:
        try:
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            p_opt, score = _local_optimize_params(params, maxiter_nm=200, maxiter_lbfgs=200)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Final polish with higher beta
    if best_params is not None and time.time() < DEADLINE - 5:
        try:
            res_final = minimize(
                lambda p: _soft_objective(p, beta=200.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            exact_s = _exact_objective(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    return _reconstruct(best_params)


if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
