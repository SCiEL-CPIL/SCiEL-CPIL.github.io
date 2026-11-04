import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _project_to_bary_simplex(bary):
    """Project onto the barycentric simplex: u>=0, v>=0, u+v<=1."""
    bary = np.array(bary, dtype=float)
    bary = np.maximum(bary, 0.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_dets_from_bary(bary):
    """Compute all triple determinants (2*area) from barycentric coords."""
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    du_ji = uj - ui
    dv_ji = vj - vi
    du_ki = uk - ui
    dv_ki = vk - vi
    
    dets = np.sqrt(3) / 2.0 * (du_ji * dv_ki - dv_ji * du_ki)
    return dets

def _min_area_ratio_from_bary(bary):
    """Compute the minimum area (divided by triangle area) from barycentric coords."""
    dets = _all_dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    return 0.5 * min_det / TRI_AREA

def _objective_smooth(x, beta=50.0):
    """Smooth objective: negative log-sum-exp of negative areas."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    lse = max_nb + np.log(np.sum(np.exp(neg_beta_dets - max_nb)))
    return lse / beta

def _objective_grad(x, beta=50.0):
    """Objective with gradient for L-BFGS-B."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0, np.zeros_like(x)
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    exp_vals = np.exp(neg_beta_dets - max_nb)
    sum_exp = np.sum(exp_vals)
    lse = max_nb + np.log(sum_exp)
    val = lse / beta
    
    idx = _IDX_TRIPLES
    signs = np.sign(dets)
    
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    sqrt3_2 = np.sqrt(3) / 2.0
    
    grad_det = np.zeros((N_TRIPLES, N, 2))
    
    grad_det[:, idx[:, 0], 0] = sqrt3_2 * (vj - vk)
    grad_det[:, idx[:, 0], 1] = sqrt3_2 * (uj - uk)
    grad_det[:, idx[:, 1], 0] = sqrt3_2 * (vk - vi)
    grad_det[:, idx[:, 1], 1] = -sqrt3_2 * (uk - ui)
    grad_det[:, idx[:, 2], 0] = -sqrt3_2 * (vj - vi)
    grad_det[:, idx[:, 2], 1] = sqrt3_2 * (uj - ui)
    
    sign_det = signs[:, None, None]
    grad_abs_det = sign_det * grad_det
    
    weights = exp_vals / sum_exp
    
    grad_bary = -np.sum(weights[:, None, None] * grad_abs_det, axis=0)
    
    grad_x = np.zeros_like(x)
    grad_x[:N*2] = grad_bary.flatten()
    
    return val, grad_x

def _objective_slsqp(x):
    """Objective for SLSQP (uses exact min area)."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    min_abs_det = np.min(np.abs(dets))
    return -min_abs_det

def _structured_init_fps():
    """Farthest-point-sampling structured initialization in barycentric coords."""
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    points.append(np.array([1.0/3, 1.0/3]))
    
    rng = np.random.RandomState(789)
    
    while len(points) < N:
        candidates = []
        for _ in range(2000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v <= 1.0:
                candidates.append(np.array([u, v]))
        candidates = np.array(candidates)
        
        best_candidate = None
        best_min_dist = -1
        
        for cand in candidates:
            dists = [np.linalg.norm(cand - p) for p in points]
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = cand
        
        if best_candidate is not None:
            points.append(best_candidate)
    
    return np.array(points)

def _simulated_annealing(bary_init, time_budget=10.0, rng=None):
    """
    Simulated Annealing to escape local minima.
    Moves points randomly and accepts worse solutions with probability exp(-delta/T).
    """
    if rng is None:
        rng = np.random.RandomState(123)
        
    bary = bary_init.copy()
    score = _min_area_ratio_from_bary(bary)
    
    T_start = 0.05
    T_end = 0.001
    steps = 5000
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Precompute indices for fast area calculation
    idx = _IDX_TRIPLES
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Annealing schedule
        alpha = i / steps
        T = T_start * (T_end / T_start) ** alpha
        
        # Pick a random point to move
        p_idx = rng.randint(N)
        
        # Random step size (Gaussian)
        step_size = 0.05 * (1.0 - alpha * 0.5)
        delta = rng.normal(0, step_size, size=2)
        
        new_bary = bary.copy()
        new_bary[p_idx] += delta
        new_bary = _project_to_bary_simplex(new_bary)
        
        new_score = _min_area_ratio_from_bary(new_bary)
        
        delta_score = new_score - score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            bary = new_bary
            score = new_score
            
    return bary, score

def _local_optimize_nm(x0, maxiter=500, xatol=1e-9, fatol=1e-9):
    """Nelder-Mead local optimization."""
    try:
        res = minimize(
            _objective_smooth, x0, method='Nelder-Mead',
            args=(30.0,),
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=300, gtol=1e-8):
    """L-BFGS-B with gradient."""
    try:
        res = minimize(
            _objective_grad, x0, method='L-BFGS-B',
            jac=True,
            args=(30.0,),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """SLSQP with simplex constraints."""
    n = len(x0) // 2
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * n)
    
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _final_polish(bary_best, time_budget=10.0):
    """High-precision final polish using SLSQP + Nelder-Mead with tight tolerances."""
    start = time.time()
    deadline = start + time_budget
    
    bary = bary_best.copy()
    score = _min_area_ratio_from_bary(bary)
    
    # SLSQP with very tight tolerance
    if time.time() < deadline - 3.0:
        x_sl = _local_optimize_slsqp(bary.flatten(), maxiter=300)
        bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
        score_sl = _min_area_ratio_from_bary(bary_sl)
        if score_sl > score:
            score = score_sl
            bary = bary_sl.copy()
    
    # Nelder-Mead with very tight tolerance
    if time.time() < deadline - 1.0:
        x_nm = _local_optimize_nm(bary.flatten(), maxiter=1000, xatol=1e-12, fatol=1e-12)
        bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
        score_nm = _min_area_ratio_from_bary(bary_nm)
        if score_nm > score:
            score = score_nm
            bary = bary_nm.copy()
    
    # Another SLSQP pass on the result
    if time.time() < deadline - 0.5:
        x_sl2 = _local_optimize_slsqp(bary.flatten(), maxiter=200)
        bary_sl2 = _project_to_bary_simplex(x_sl2.reshape(-1, 2))
        score_sl2 = _min_area_ratio_from_bary(bary_sl2)
        if score_sl2 > score:
            score = score_sl2
            bary = bary_sl2.copy()
    
    return bary, score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    # Generate multiple structured starts
    starts = []
    
    # FPS-based start
    starts.append(_structured_init_fps())
    
    # Hexagonal lattice
    bary_hex = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75],
    ])
    starts.append(_project_to_bary_simplex(bary_hex.copy()))
    
    # Symmetric grid
    bary_grid = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.25, 0.0], [0.75, 0.0], [0.0, 0.25], [0.0, 0.75],
        [0.25, 0.25], [0.5, 0.25], [0.25, 0.5], [0.5, 0.5],
    ])
    starts.append(_project_to_bary_simplex(bary_grid.copy()))
    
    # Random structured
    rng = np.random.RandomState(42)
    bary_rand = []
    bary_rand.append([0.0, 0.0])
    bary_rand.append([1.0, 0.0])
    bary_rand.append([0.0, 1.0])
    bary_rand.append([1/3, 1/3])
    for _ in range(7):
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_rand.append([u, v])
        else:
            bary_rand.append([u * 0.7, v * 0.7])
    starts.append(_project_to_bary_simplex(np.array(bary_rand[:N])))
    
    # Evaluate initial starts
    for bary_start in starts:
        if time_left() < 5:
            break
        bary_clamped = _project_to_bary_simplex(bary_start)
        points = _bary_to_cart(bary_clamped)
        score = _min_area_ratio_from_bary(bary_clamped)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary_clamped.copy()
    
    # Multi-start optimization with Simulated Annealing
    max_restarts = 60
    restart = 0
    
    while restart < max_restarts and time_left() > 20.0:
        restart += 1
        
        # 1. Generate Start
        if restart % 5 == 0:
            # Random start
            bary_start = []
            for _ in range(N):
                u = rng.uniform(0.05, 0.9)
                v = rng.uniform(0.05, 0.9)
                if u + v < 0.9:
                    bary_start.append([u, v])
                else:
                    bary_start.append([u * 0.8, v * 0.8])
            bary_start = _project_to_bary_simplex(np.array(bary_start[:N]))
        elif restart % 2 == 0:
            # Perturb best
            scale = 0.05
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _project_to_bary_simplex(bary_pert)
        else:
            # Perturb best more
            scale = 0.1
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _project_to_bary_simplex(bary_pert)
        
        bary_opt = bary_start.copy()
        
        # 2. Simulated Annealing Phase (Global Exploration)
        if time_left() > 15.0:
            sa_budget = min(5.0, time_left() - 15.0)
            bary_sa, score_sa = _simulated_annealing(bary_opt, time_budget=sa_budget, rng=rng)
            if score_sa > best_score:
                best_score = score_sa
                best_points = _bary_to_cart(bary_sa).copy()
                best_bary = bary_sa.copy()
            bary_opt = bary_sa.copy()
            
        # 3. Local Optimization Phase (Exploitation)
        if time_left() > 10.0:
            # Nelder-Mead is often best for the non-smooth min-area
            x_nm = _local_optimize_nm(bary_opt.flatten(), maxiter=200)
            bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
            score_nm = _min_area_ratio_from_bary(bary_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = _bary_to_cart(bary_nm).copy()
                best_bary = bary_nm.copy()
            bary_opt = bary_nm.copy()
            
        if time_left() > 8.0:
            # SLSQP for final constraint handling
            x_sl = _local_optimize_slsqp(bary_opt.flatten(), maxiter=100)
            bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
            score_sl = _min_area_ratio_from_bary(bary_sl)
            if score_sl > best_score:
                best_score = score_sl
                best_points = _bary_to_cart(bary_sl).copy()
                best_bary = bary_sl.copy()
    
    # Final fine-tuning with high-precision polish
    if time_left() > 8.0 and best_bary is not None:
        # Do a few rounds of small perturbation + polish
        for ft in range(8):
            if time_left() < 4.0:
                break
            bary_ft = best_bary + rng.normal(0, 0.003, size=best_bary.shape)
            bary_ft = _project_to_bary_simplex(bary_ft)
            
            x_sl = _local_optimize_slsqp(bary_ft.flatten(), maxiter=100)
            bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
            score_sl = _min_area_ratio_from_bary(bary_sl)
            
            if score_sl > best_score:
                best_score = score_sl
                best_points = _bary_to_cart(bary_sl).copy()
                best_bary = bary_sl.copy()
            
            x_nm = _local_optimize_nm(bary_sl.flatten(), maxiter=200, xatol=1e-10, fatol=1e-10)
            bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
            score_nm = _min_area_ratio_from_bary(bary_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_points = _bary_to_cart(bary_nm).copy()
                best_bary = bary_nm.copy()
        
        # Apply the high-precision final polish
        if time_left() > 5.0:
            bary_polished, score_polished = _final_polish(best_bary, time_budget=min(10.0, time_left() - 3.0))
            if score_polished > best_score:
                best_score = score_polished
                best_points = _bary_to_cart(bary_polished).copy()
                best_bary = bary_polished.copy()
    
    # Fallback
    if best_points is None:
        bary_fb = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75],
        ])
        best_points = _bary_to_cart(_project_to_bary_simplex(bary_fb))
    
    return best_points
