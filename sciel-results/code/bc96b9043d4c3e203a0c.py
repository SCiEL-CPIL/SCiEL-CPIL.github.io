"""Rewritten packing program: soft-min objective with L-BFGS-B, multiple restarts, and polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill


def _radius(c):
    """Compute the achievable common radius for centres c (N,3) in [-0.5,0.5]^3."""
    if c.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Boundary distances
    bd = np.minimum(
        np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        ),
        np.minimum(0.5 - c[:, 2], 0.5 + c[:, 2])
    )
    min_bd = np.min(bd)
    if min_bd <= 0:
        return 0.0
    # Pairwise distances
    # Use broadcasting for 100x100
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_bd, min_pair)


def _soft_min_obj(x, c, temperature):
    """
    Objective: -soft_min of all constraint margins.
    x is flattened (3N,), c is initial centres for reference.
    Constraints: for each pair (i,j), margin_ij = |c_i - c_j| - 2r_target
                 for each centre i, margin_i = 0.5 - |c_i|_inf  (distance to wall)
    We want to maximize min(all margins). Use soft-min via log-sum-exp.
    
    Actually, let's reformulate: we want to maximize r such that all pair distances >= 2r 
    and all wall distances >= r.
    
    Alternative: directly optimize r as a variable? No, let's use the approach:
    Define f(x) = soft_min of { d_ij/2 for all pairs, w_i for all walls }
    where d_ij = |c_i - c_j|, w_i = min_k(0.5 - |c_ik|)
    
    We maximize f(x). Use negative for minimization.
    """
    c = x.reshape(-1, 3)
    
    # Wall margins
    walls = np.minimum(
        np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        ),
        np.minimum(0.5 - c[:, 2], 0.5 + c[:, 2])
    )
    
    # Pair margins (half distances)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pairs = dist / 2.0
    pairs = pairs[np.triu_indices(N, k=1)]
    
    # Combine all margins
    all_margins = np.concatenate([walls, pairs])
    
    # Soft min via log-sum-exp (temperature controls smoothness)
    # soft_min = -t * log(sum(exp(-m_i / t)))
    # For numerical stability:
    m_min = np.min(all_margins)
    exp_vals = np.exp(-(all_margins - m_min) / temperature)
    soft_min = m_min - temperature * np.log(np.sum(exp_vals))
    
    return -soft_min  # minimize negative


def _soft_min_grad(x, temperature):
    """Gradient of the soft-min objective."""
    c = x.reshape(-1, 3)
    
    # Wall margins
    walls = np.minimum(
        np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        ),
        np.minimum(0.5 - c[:, 2], 0.5 + c[:, 2])
    )
    
    # Pair margins
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    np.fill_diagonal(dist, np.inf)
    pairs_full = dist / 2.0
    pairs = pairs_full[np.triu_indices(N, k=1)]
    
    all_margins = np.concatenate([walls, pairs])
    
    m_min = np.min(all_margins)
    exp_vals = np.exp(-(all_margins - m_min) / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of soft_min w.r.t. margins: -weights
    # soft_min = m_min - t * log(sum_exp)
    # d(soft_min)/d(m_i) = -weights_i
    
    # Now compute gradient w.r.t. x
    grad = np.zeros_like(x)
    
    # Wall contribution
    # walls_i = min over coords of (0.5 - |c_ik|)
    # For each wall margin, find which coordinate is active
    for i in range(N):
        c_i = c[i]
        # Compute individual wall distances
        wd = np.array([
            min(0.5 - c_i[0], 0.5 + c_i[0]),
            min(0.5 - c_i[1], 0.5 + c_i[1]),
            min(0.5 - c_i[2], 0.5 + c_i[2])
        ])
        active = np.argmin(wd)
        # Gradient of wall distance w.r.t. c_i[active]
        coord = c_i[active]
        if coord >= 0:
            # 0.5 - coord, grad = -1
            g = -1.0
        else:
            # 0.5 + coord, grad = +1
            g = 1.0
        grad[3*i + active] += weights[i] * g  # d(soft_min)/d(wall_i) * d(wall_i)/d(x)
        # But we have -soft_min, so flip sign
        # Actually, objective is -soft_min, so grad_obj = -grad_soft_min
    
    # This is getting complicated. Let's use numerical gradient or a simpler approach.
    # Given time constraints, let's use a simpler optimization approach.
    return None


def _construct_grid_init(m=None):
    """Generate a grid initialization."""
    if m is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _construct_bcc_init():
    """Generate a BCC-like initialization."""
    # 4x4x4 = 64 grid points, plus 36 more
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add BCC center points
    bcc_pts = []
    for i in range(m-1):
        for j in range(m-1):
            for k in range(m-1):
                cx = (ticks[i] + ticks[i+1]) / 2
                cy = (ticks[j] + ticks[j+1]) / 2
                cz = (ticks[k] + ticks[k+1]) / 2
                bcc_pts.append((cx, cy, cz))
    
    all_pts = np.concatenate([grid, np.array(bcc_pts)], axis=0)
    # Shuffle and take first N
    idx = np.random.choice(len(all_pts), N, replace=False)
    return all_pts[idx]


def construct():
    start_time = time.time()
    best_c = _construct_grid_init()
    best_r = _radius(best_c)
    
    # Try multiple initializations
    inits = []
    inits.append(('grid', _construct_grid_init()))
    inits.append(('bcc', _construct_bcc_init()))
    
    # Random perturbations
    for i in range(3):
        base = _construct_grid_init()
        noise = np.random.randn(N, 3) * 0.02
        perturbed = base + noise
        # Clip to stay in cube
        perturbed = np.clip(perturbed, -0.48, 0.48)
        inits.append((f'rand{i}', perturbed))
    
    # Optimize each initialization
    for init_name, init_c in inits:
        if time.time() - start_time > DEADLINE:
            break
        
        x0 = init_c.flatten()
        
        # Use Nelder-Mead for the first few iterations to handle non-smoothness,
        # then switch to L-BFGS-B
        try:
            # Phase 1: Nelder-Mead
            res_nm = minimize(
                _soft_min_obj, x0, args=(init_c, 0.01),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if time.time() - start_time > DEADLINE:
                # Evaluate current best
                c_try = res_nm.x.reshape(-1, 3)
                r_try = _radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try
                continue
            
            # Phase 2: L-BFGS-B with tighter temperature
            x1 = res_nm.x
            res_lbfgs = minimize(
                _soft_min_obj, x1, args=(init_c, 0.005),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            c_try = res_lbfgs.x.reshape(-1, 3)
            r_try = _radius(c_try)
            
            if r_try > best_r:
                best_r = r_try
                best_c = c_try
                
        except Exception:
            continue
    
    # Polish: try to improve the best solution with a very fine search
    if time.time() - start_time < DEADLINE - 10:
        # Local perturbation search
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Small random perturbation
            noise = np.random.randn(N, 3) * 0.005
            c_perturbed = best_c + noise
            c_perturbed = np.clip(c_perturbed, -0.49, 0.49)
            
            x0 = c_perturbed.flatten()
            try:
                res = minimize(
                    _soft_min_obj, x0, args=(best_c, 0.002),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
                )
                c_try = res.x.reshape(-1, 3)
                r_try = _radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try
            except Exception:
                continue
    
    return best_c
