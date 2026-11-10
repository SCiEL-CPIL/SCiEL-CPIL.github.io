# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(pts):
    """Return (d_min/d_max)^2, or 0 on degenerate input."""
    try:
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        dmin = d.min()
        dmax = d.max()
        if dmax <= 1e-12 or not np.isfinite(dmin) or not np.isfinite(dmax):
            return 0.0
        r = dmin / dmax
        return float(r * r)
    except Exception:
        return 0.0

def _build_points(params):
    """
    Build 14 points from 5 parameters:
    z_p: z-coordinate of pole 1 (pole 2 is -z_p)
    r_1, z_1: radius and z-height of ring 1 (6 points, phase 0)
    r_2, z_2: radius and z-height of ring 2 (6 points, phase pi/6)
    """
    z_p, r_1, z_1, r_2, z_2 = params
    points = np.zeros((14, 3))
    
    # Poles
    points[0] = [0.0, 0.0, z_p]
    points[1] = [0.0, 0.0, -z_p]
    
    # Ring 1: 6 points, phase 0
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for i in range(6):
        points[2 + i] = [r_1 * np.cos(angles1[i]), r_1 * np.sin(angles1[i]), z_1]
        
    # Ring 2: 6 points, phase pi/6 (staggered)
    angles2 = angles1 + np.pi / 6.0
    for i in range(6):
        points[8 + i] = [r_2 * np.cos(angles2[i]), r_2 * np.sin(angles2[i]), z_2]
        
    return points

def _neg_score_params(params):
    """Negative score for minimization."""
    pts = _build_points(params)
    return -_score(pts)

def _smooth_score(pts, beta=50.0):
    """
    Smooth surrogate: use soft-min and soft-max of pairwise distances.
    soft_min = -1/beta * log(sum(exp(-beta*d)))
    soft_max = 1/beta * log(sum(exp(beta*d)))
    ratio = soft_min / soft_max
    return ratio^2
    """
    try:
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        # Clip to avoid overflow
        d_clip = np.clip(d, 1e-10, None)
        # Soft min
        neg_beta_d = -beta * d_clip
        neg_beta_d -= neg_beta_d.max()  # numerical stability
        soft_min = -1.0 / beta * np.log(np.sum(np.exp(neg_beta_d)))
        # Soft max
        beta_d = beta * d_clip
        beta_d -= beta_d.max()
        soft_max = 1.0 / beta * np.log(np.sum(np.exp(beta_d)))
        if soft_max <= 1e-12:
            return 0.0
        r = soft_min / soft_max
        return float(r * r)
    except Exception:
        return 0.0

def _neg_smooth_score_params(params, beta=50.0):
    pts = _build_points(params)
    return -_smooth_score(pts, beta)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin (600s limit, 10s buffer)
    
    best_pts = None
    best_val = -1.0
    
    def _update_best(pts, val):
        nonlocal best_pts, best_val
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # Stage 1: Coarse grid search
    z_p_vals = np.linspace(0.0, 1.2, 13)
    r_1_vals = np.linspace(0.2, 1.2, 11)
    z_1_vals = np.linspace(-0.8, 0.8, 17)
    r_2_vals = np.linspace(0.2, 1.2, 11)
    z_2_vals = np.linspace(-0.8, 0.8, 17)
    
    ZP, R1, Z1, R2, Z2 = np.meshgrid(z_p_vals, r_1_vals, z_1_vals, r_2_vals, z_2_vals, indexing='ij')
    grid_params = np.column_stack([ZP.ravel(), R1.ravel(), Z1.ravel(), R2.ravel(), Z2.ravel()])
    
    best_param_coarse = None
    best_score_coarse = -1.0
    
    chunk_size = 5000
    total_params = len(grid_params)
    for i in range(0, total_params, chunk_size):
        if time.time() > deadline - 120:
            break
        chunk = grid_params[i:i+chunk_size]
        for j in range(len(chunk)):
            pts = _build_points(chunk[j])
            s = _score(pts)
            if s > best_score_coarse:
                best_score_coarse = s
                best_param_coarse = chunk[j].copy()
    
    if best_param_coarse is None:
        best_param_coarse = np.array([0.5, 0.8, 0.0, 0.8, 0.0])
        best_score_coarse = _score(_build_points(best_param_coarse))
    
    _update_best(_build_points(best_param_coarse), best_score_coarse)
    
    # Stage 2: Fine grid around best coarse candidate
    z_p_min = max(0.0, best_param_coarse[0] - 0.15)
    z_p_max = min(1.5, best_param_coarse[0] + 0.15)
    r_1_min = max(0.1, best_param_coarse[1] - 0.15)
    r_1_max = min(1.5, best_param_coarse[1] + 0.15)
    z_1_min = max(-1.5, best_param_coarse[2] - 0.15)
    z_1_max = min(1.5, best_param_coarse[2] + 0.15)
    r_2_min = max(0.1, best_param_coarse[3] - 0.15)
    r_2_max = min(1.5, best_param_coarse[3] + 0.15)
    z_2_min = max(-1.5, best_param_coarse[4] - 0.15)
    z_2_max = min(1.5, best_param_coarse[4] + 0.15)
    
    n_fine = 30
    z_p_fine = np.linspace(z_p_min, z_p_max, n_fine)
    r_1_fine = np.linspace(r_1_min, r_1_max, n_fine)
    z_1_fine = np.linspace(z_1_min, z_1_max, n_fine)
    r_2_fine = np.linspace(r_2_min, r_2_max, n_fine)
    z_2_fine = np.linspace(z_2_min, z_2_max, n_fine)
    
    ZPF, R1F, Z1F, R2F, Z2F = np.meshgrid(z_p_fine, r_1_fine, z_1_fine, r_2_fine, z_2_fine, indexing='ij')
    fine_params = np.column_stack([ZPF.ravel(), R1F.ravel(), Z1F.ravel(), R2F.ravel(), Z2F.ravel()])
    
    best_param_fine = best_param_coarse.copy()
    best_score_fine = best_score_coarse
    
    total_fine = len(fine_params)
    for i in range(0, total_fine, chunk_size):
        if time.time() > deadline - 90:
            break
        chunk = fine_params[i:i+chunk_size]
        for j in range(len(chunk)):
            pts = _build_points(chunk[j])
            s = _score(pts)
            if s > best_score_fine:
                best_score_fine = s
                best_param_fine = chunk[j].copy()
    
    _update_best(_build_points(best_param_fine), best_score_fine)
    
    # Stage 3: Smooth optimization with annealed beta
    # Use soft-min/soft-max surrogate to get a smooth landscape
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(
                lambda p: _neg_smooth_score_params(p, beta),
                best_param_fine,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = _score(_build_points(res.x))
            if s > best_score_fine:
                best_score_fine = s
                best_param_fine = res.x.copy()
            _update_best(_build_points(best_param_fine), best_score_fine)
        except Exception:
            pass
    
    # Stage 4: Exact local optimization (non-smooth)
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _neg_score_params,
                best_param_fine,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            s = _score(_build_points(res.x))
            if s > best_score_fine:
                best_score_fine = s
                best_param_fine = res.x.copy()
            _update_best(_build_points(best_param_fine), best_score_fine)
        except Exception:
            pass
    
    # Stage 5: Powell polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_score_params,
                best_param_fine,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            s = _score(_build_points(res.x))
            if s > best_score_fine:
                best_score_fine = s
                best_param_fine = res.x.copy()
            _update_best(_build_points(best_param_fine), best_score_fine)
        except Exception:
            pass
    
    # Stage 6: Multiple restarts with random perturbations around best
    if time.time() < deadline - 5:
        rng = np.random.default_rng(12345)
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            perturb = rng.normal(0, 0.05, 5)
            perturbed = best_param_fine + perturb
            # Clip to valid ranges
            perturbed[0] = np.clip(perturbed[0], 0.0, 1.5)
            perturbed[1] = np.clip(perturbed[1], 0.1, 1.5)
            perturbed[2] = np.clip(perturbed[2], -1.5, 1.5)
            perturbed[3] = np.clip(perturbed[3], 0.1, 1.5)
            perturbed[4] = np.clip(perturbed[4], -1.5, 1.5)
            try:
                res = minimize(
                    _neg_score_params,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = _score(_build_points(res.x))
                if s > best_score_fine:
                    best_score_fine = s
                    best_param_fine = res.x.copy()
                _update_best(_build_points(best_param_fine), best_score_fine)
            except Exception:
                pass
    
    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _build_points(np.array([0.5, 0.8, 0.0, 0.8, 0.0]))
    
    return best_pts

# EVOLVE-BLOCK-END
