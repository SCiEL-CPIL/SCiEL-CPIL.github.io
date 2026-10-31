import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

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
    Soft min/max surrogate for gradient-based optimization.
    Uses log-sum-exp for numerical stability.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max via log-sum-exp
    max_d_sq = np.max(d_sq)
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min
    min_d_sq = np.min(d_sq)
    soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _soft_objective_grad(params, beta):
    """
    Gradient of soft min/max surrogate.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq_iu = d_sq[iu]
    d_sq_iu = np.maximum(d_sq_iu, 1e-12)
    
    max_d_sq = np.max(d_sq_iu)
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq_iu - max_d_sq))))
    
    min_d_sq = np.min(d_sq_iu)
    soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq_iu - min_d_sq))))
    
    if soft_min < 1e-8:
        return 1e10, np.zeros_like(params)
    
    obj = soft_max / soft_min
    
    # Gradient wrt d_sq_iu
    exp_max = np.exp(beta * (d_sq_iu - max_d_sq))
    grad_soft_max = exp_max / np.sum(exp_max)
    
    exp_min = np.exp(-beta * (d_sq_iu - min_d_sq))
    grad_soft_min = -exp_min / np.sum(exp_min)
    
    grad_obj_dsq = (grad_soft_max * soft_min - soft_max * grad_soft_min) / (soft_min ** 2)
    
    # Gradient wrt P: d(d_sq)/dP_i = 2*(P_i - P_j) for pair (i,j)
    grad_P = np.zeros((N, 2))
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        g = 2.0 * grad_obj_dsq[k] * (P[i] - P[j])
        grad_P[i] += g
        grad_P[j] -= g
    
    # Only return gradient for params (points 2 onward)
    grad_params = grad_P[2:].flatten()
    
    return obj, grad_params

def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
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

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset - compact triangular cluster
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
    
    # Jittered hex with multiple sigmas
    for sigma in [0.03, 0.08, 0.15, 0.25]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.05, (16, 2)))
    candidates.append(grid + rng.normal(0, 0.15, (16, 2)))
    
    # 3. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    candidates.append(np.vstack([inner, outer]) + rng.normal(0, 0.05, (16, 2)))
    
    # 4. Golden angle spiral (Fibonacci)
    n = 16
    golden_angle = np.pi * (3. - np.sqrt(5.))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(n)) / np.sqrt(n) * 2.0
    spiral = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(spiral)
    candidates.append(spiral + rng.normal(0, 0.05, (16, 2)))
    candidates.append(spiral + rng.normal(0, 0.15, (16, 2)))
    
    # 5. Random uniform in disk
    for _ in range(6):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Random on ring with varying radius
    for _ in range(4):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 7. Two concentric rings with offset
    for r1, r2, offset in [(0.4, 1.2, 0), (0.6, 1.4, np.pi/8), (0.5, 1.3, np.pi/16)]:
        angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + offset
        inner = np.column_stack([r1 * np.cos(angles_in), r1 * np.sin(angles_in)])
        outer = np.column_stack([r2 * np.cos(angles_out), r2 * np.sin(angles_out)])
        candidates.append(np.vstack([inner, outer]))
    
    return candidates

def _local_optimize(params, maxiter_nm=300, maxiter_lbfgs=500, deadline=None):
    """
    Two-phase local optimization:
    1. Nelder-Mead on exact objective
    2. L-BFGS-B on soft objective with annealing (with gradient)
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    if deadline is not None and time.time() > deadline - 5:
        return best_params, best_score
    
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
    
    if deadline is not None and time.time() > deadline - 5:
        return best_params, best_score
    
    # Phase 2: L-BFGS-B with annealing on soft objective (with analytical gradient)
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5:
            break
        try:
            def obj_and_grad(p):
                return _soft_objective_grad(p, beta)
            
            res_lbfgs = minimize(
                obj_and_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
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
    deadline = start_time + 135  # 15s safety margin
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Local optimization from diverse structured starts
    for P0 in candidates:
        if time.time() > deadline - 30:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            p_opt, score = _local_optimize(params, maxiter_nm=200, maxiter_lbfgs=200, deadline=deadline)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: More random restarts in reduced space
    restart_count = 0
    while time.time() < deadline - 25:
        try:
            # Generate random params in reduced space
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            p_opt, score = _local_optimize(params, maxiter_nm=150, maxiter_lbfgs=150, deadline=deadline)
            if score < best_score:
                best_score = score
                best_params = p_opt
            restart_count += 1
        except Exception:
            pass
    
    # Phase 3: Perturbation-based refinement around best solution
    if best_params is not None and time.time() < deadline - 15:
        for scale in [0.01, 0.005, 0.001]:
            if time.time() > deadline - 10:
                break
            try:
                perturbed = best_params + rng.normal(0, scale, (14, 2)).flatten()
                p_opt, score = _local_optimize(perturbed, maxiter_nm=100, maxiter_lbfgs=100, deadline=deadline)
                if score < best_score:
                    best_score = score
                    best_params = p_opt
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with gradient
    if best_params is not None and time.time() < deadline - 5:
        for beta in [5000.0, 10000.0]:
            if time.time() > deadline - 3:
                break
            try:
                def obj_and_grad_final(p):
                    return _soft_objective_grad(p, beta)
                
                res_final = minimize(
                    obj_and_grad_final,
                    best_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
                )
                exact_s = _exact_objective(res_final.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = res_final.x.copy()
            except Exception:
                pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback
        best_params = _prepare_params(candidates[0])
        if best_params is None:
            return candidates[0]
    
    P_final = np.zeros((N, 2))
    P_final[0] = [0.0, 0.0]
    P_final[1] = [1.0, 0.0]
    P_final[2:] = best_params.reshape(-1, 2)
    return P_final

# EVOLVE-BLOCK-END
