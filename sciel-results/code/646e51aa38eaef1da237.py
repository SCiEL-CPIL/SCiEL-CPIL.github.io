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
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R.T / norm_v
    
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

def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = _reconstruct(params)
    return _score(P)

def _soft_objective(params, temp=0.01):
    """
    Soft min/max surrogate using log-sum-exp for smooth gradients.
    temp controls sharpness: smaller = closer to exact min/max.
    """
    P = _reconstruct(params)
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    m_max = np.max(d_sq)
    soft_max = m_max + temp * np.log(np.sum(np.exp((d_sq - m_max) / temp)))
    
    m_min = np.min(d_sq)
    soft_min = m_min - temp * np.log(np.sum(np.exp(-(d_sq - m_min) / temp)))
    
    if soft_min < 1e-8:
        return 1e10
    return soft_max / soft_min

def _epigraph_objective(params, temp=0.01):
    """
    Epigraph formulation: minimize t = soft_max / soft_min,
    but with an explicit auxiliary variable handled via penalty.
    This is equivalent to the soft objective but structured differently
    to allow tighter convergence in some cases.
    """
    P = _reconstruct(params)
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    m_max = np.max(d_sq)
    soft_max = m_max + temp * np.log(np.sum(np.exp((d_sq - m_max) / temp)))
    
    m_min = np.min(d_sq)
    soft_min = m_min - temp * np.log(np.sum(np.exp(-(d_sq - m_min) / temp)))
    
    if soft_min < 1e-8:
        return 1e10
    return soft_max / soft_min

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset
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
    
    # Jittered hex
    for sigma in [0.05, 0.1, 0.2]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid = np.column_stack([X.flatten(), Y.flatten()])
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 2b. Compressed 4x4 grid (corners pulled toward center)
    comp_pts = grid.copy()
    center = np.array([1.5, 1.5])
    for idx in [0, 3, 12, 15]:
        comp_pts[idx] = comp_pts[idx] + 0.3 * (center - comp_pts[idx])
    candidates.append(comp_pts)
    
    # 3. Triangular lattice rhombus (from inspiration 1)
    tri_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2], [2.5, np.sqrt(3)/2], [3.5, np.sqrt(3)/2],
        [1, np.sqrt(3)], [2, np.sqrt(3)], [3, np.sqrt(3)], [4, np.sqrt(3)],
        [1.5, 1.5*np.sqrt(3)], [2.5, 1.5*np.sqrt(3)], [3.5, 1.5*np.sqrt(3)], [4.5, 1.5*np.sqrt(3)]
    ], dtype=float)
    candidates.append(tri_pts)
    candidates.append(tri_pts + rng.normal(0, 0.05, (16, 2)))
    
    # 4. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 4b. Optimized concentric rings (multiple radii/phases)
    for r1, r2, phase in [(0.4, 1.2, 0), (0.5, 1.4, np.pi/8), (0.6, 1.6, np.pi/4), (0.45, 1.3, np.pi/12)]:
        angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
        angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase + np.pi/8
        inner = np.column_stack([r1 * np.cos(angles_in), r1 * np.sin(angles_in)])
        outer = np.column_stack([r2 * np.cos(angles_out), r2 * np.sin(angles_out)])
        candidates.append(np.vstack([inner, outer]))
    
    # 5. Random uniform
    for _ in range(5):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 6. Random on ring
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 7. Square-root radial grid
    base_grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    r = np.sqrt(np.sum(base_grid**2, axis=1))
    scaled = base_grid * (1.0 + 0.3 * np.sqrt(r)[:, np.newaxis])
    candidates.append(scaled)
    
    # 8. Golden angle spiral (from inspiration 2)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    spiral = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i + 1)
        theta = i * golden_angle
        spiral[i, 0] = r * np.cos(theta)
        spiral[i, 1] = r * np.sin(theta)
    candidates.append(spiral)
    
    return candidates

def _annealed_refine(params, t0, deadline, max_total_iters=5000):
    """
    Annealing schedule on the soft objective in reduced parameter space.
    Starts with high temperature (broad search), decreases to very low
    (fine polish). Uses L-BFGS-B at each temperature stage.
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Annealing schedule: progressively sharper
    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    current_params = best_params.copy()
    
    for temp in temps:
        if time.time() - t0 > deadline - 5:
            break
        
        try:
            res = minimize(
                lambda p, t=temp: _soft_objective(p, t),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': max(200, max_total_iters // len(temps)),
                         'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            exact_s = _exact_objective(res.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res.x.copy()
                current_params = res.x.copy()
            else:
                # Keep current if it didn't improve (annealing step)
                current_params = res.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def _local_optimize_params(params, t0, deadline):
    """
    Multi-stage local optimization in reduced parameter space:
    1. Nelder-Mead on exact objective (coarse)
    2. Annealed L-BFGS-B on soft objective (refinement)
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Nelder-Mead on exact objective for coarse local search
    if time.time() - t0 < deadline - 15:
        try:
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 2: Annealed L-BFGS-B on soft objective
    if time.time() - t0 < deadline - 5:
        p_opt, score = _annealed_refine(best_params, t0, deadline, max_total_iters=4000)
        if score < best_score:
            best_score = score
            best_params = p_opt
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
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
            
            p_opt, score = _local_optimize_params(params, start_time, deadline)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Random restarts in reduced space
    while time.time() < deadline - 30:
        try:
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            p_opt, score = _local_optimize_params(params, start_time, deadline)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with very low temperature
    if best_params is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                lambda p: _soft_objective(p, temp=0.0001),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            exact_s = _exact_objective(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more Nelder-Mead polish on exact objective
    if best_params is not None and time.time() < deadline - 1:
        try:
            res_nm_final = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm_final.fun < best_score:
                best_score = res_nm_final.fun
                best_params = res_nm_final.x.copy()
        except Exception:
            pass
    
    # Reconstruct final points
    if best_params is None:
        best_params = _prepare_params(candidates[0])
        if best_params is None:
            return candidates[0]
    
    P_final = _reconstruct(best_params)
    return P_final

# EVOLVE-BLOCK-END
