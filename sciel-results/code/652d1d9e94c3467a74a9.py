"""Combined program: Smooth Log-Sum-Exp with Penalty Polish for 100 Spheres."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_smooth_obj_grad(c, alpha):
    """
    Compute the objective value and gradient for the smooth log-sum-exp approximation
    of the negative packing radius.
    """
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    exp_terms = np.exp(-alpha * all_dists)
    
    Z = np.sum(exp_terms)
    if Z == 0:
        return 0.0, np.zeros_like(c)
        
    obj_val = (1.0 / alpha) * np.log(Z)
    
    grad_c = np.zeros_like(c)
    
    exp_pair = exp_terms[:len(pair_dists)]
    weights_pair = exp_pair / Z
    
    pair_dists_raw = d[i_idx, j_idx]
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms
    
    w = weights_pair[:, np.newaxis]
    contrib = w * u / 2.0
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    exp_wall = exp_terms[len(pair_dists):]
    weights_wall = exp_wall / Z
    ww = weights_wall.reshape(6, N)
    
    grad_c[:, 0] += ww[0] - ww[1]
    grad_c[:, 1] += ww[2] - ww[3]
    grad_c[:, 2] += ww[4] - ww[5]
    
    return obj_val, grad_c


def _compute_penalty_obj_grad(c, mu):
    """
    Penalty-based objective for final polish.
    Maximizes min distance by penalizing violations of a target radius.
    """
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx]
    
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])
    
    all_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Estimate current radius
    r_est = np.min(all_dists)
    
    # Penalty: sum of squared violations below target
    target = r_est + 0.001  # Push slightly above current min
    violations = np.maximum(0, target - all_dists)
    
    obj_val = np.sum(violations ** 2) + 0.1 * r_est  # Small bonus for radius
    
    # Gradient
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    viol_pair = violations[:len(pair_dists)]
    active_pair = viol_pair > 1e-12
    if np.any(active_pair):
        pair_dists_raw = d[i_idx, j_idx]
        pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
        
        ci = c[i_idx[active_pair]]
        cj = c[j_idx[active_pair]]
        diff = ci - cj
        norms = pair_dists_raw_safe[active_pair, np.newaxis]
        u = diff / norms
        
        # d(violation^2)/d(c_i) = -2 * viol * u / 2
        # d(violation^2)/d(c_j) = 2 * viol * u / 2
        w = 2.0 * viol_pair[active_pair][:, np.newaxis]
        
        grad_c_idx_i = i_idx[active_pair]
        grad_c_idx_j = j_idx[active_pair]
        
        np.add.at(grad_c, grad_c_idx_i, -w * u / 2.0)
        np.add.at(grad_c, grad_c_idx_j, w * u / 2.0)
    
    # Wall part
    viol_wall = violations[len(pair_dists):]
    ww = viol_wall.reshape(6, N)
    
    # For each wall, gradient contribution
    # 0.5 - c[:, 0]: grad = -1, so d(viol^2)/dc = -2*viol*(-1) = 2*viol
    # 0.5 + c[:, 0]: grad = +1, so d(viol^2)/dc = -2*viol*(1) = -2*viol
    grad_c[:, 0] += 2.0 * ww[0] - 2.0 * ww[1]
    grad_c[:, 1] += 2.0 * ww[2] - 2.0 * ww[3]
    grad_c[:, 2] += 2.0 * ww[4] - 2.0 * ww[5]
    
    # Add gradient for the 0.1 * r_est term (approximate)
    # This is tricky, so we'll skip it for simplicity
    
    return obj_val, grad_c


def _score(c):
    """Compute the exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))


def _initial_grid():
    """Create a 5x5x4 grid."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _initial_grid_v2():
    """Create a 5x5x4 grid with offset alternating layers."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    centres = []
    for idx, z in enumerate(zs):
        offset = (0.1 if idx % 2 == 1 else 0.0)
        for y in ys:
            for x in xs:
                centres.append([x + offset, y, z])
    return np.array(centres[:N])


def _initial_grid_v3():
    """Try a different grid: 4x5x5 = 100."""
    xs = np.linspace(-0.375, 0.375, 4)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.4, 0.4, 5)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _local_optimize_smooth(c, alpha, maxiter=200, ftol=1e-12, gtol=1e-8):
    """Run L-BFGS-B optimization on the smooth objective."""
    def obj(x):
        c_flat = x.reshape(N, 3)
        val, _ = _compute_smooth_obj_grad(c_flat, alpha)
        return val
    
    def grad(x):
        c_flat = x.reshape(N, 3)
        _, g = _compute_smooth_obj_grad(c_flat, alpha)
        return g.flatten()
    
    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        result = minimize(
            obj,
            x0,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.success or result.fun < obj(x0) - 1e-12:
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            return c_new
    except Exception:
        pass
    
    return c


def _local_optimize_penalty(c, maxiter=100, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization on the penalty objective for final polish."""
    def obj(x):
        c_flat = x.reshape(N, 3)
        val, _ = _compute_penalty_obj_grad(c_flat, 1.0)
        return val
    
    def grad(x):
        c_flat = x.reshape(N, 3)
        _, g = _compute_penalty_obj_grad(c_flat, 1.0)
        return g.flatten()
    
    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        result = minimize(
            obj,
            x0,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.success or result.fun < obj(x0) - 1e-12:
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            return c_new
    except Exception:
        pass
    
    return c


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    np.random.seed(42)
    
    # Try multiple initial configurations and pick the best
    init_configs = [_initial_grid(), _initial_grid_v2(), _initial_grid_v3()]
    
    best_c = None
    best_score = 0.0
    
    for init_c in init_configs:
        score = _score(init_c)
        if score > best_score:
            best_score = score
            best_c = init_c.copy()
    
    # Phase 1: Smooth log-sum-exp annealing
    alphas = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
    
    current_alpha_idx = 0
    c = best_c.copy()
    
    while time.time() - start_time < deadline - 25.0:
        if current_alpha_idx >= len(alphas):
            alpha = alphas[-1]
        else:
            alpha = alphas[current_alpha_idx]
            current_alpha_idx += 1
            
        # Use tighter tolerances as alpha increases
        if alpha >= 500:
            ftol = 1e-14
            gtol = 1e-10
            maxiter = 300
        elif alpha >= 100:
            ftol = 1e-13
            gtol = 1e-9
            maxiter = 250
        else:
            ftol = 1e-12
            gtol = 1e-8
            maxiter = 200
        
        c_new = _local_optimize_smooth(c, alpha, maxiter=maxiter, ftol=ftol, gtol=gtol)
        
        score_new = _score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
            c = c_new.copy()
    
    # Phase 2: Penalty-based final polish
    if time.time() - start_time < deadline - 10.0:
        c_polish = best_c.copy()
        
        # Multiple rounds of penalty optimization with decreasing target
        for round_num in range(3):
            if time.time() - start_time > deadline - 10.0:
                break
            
            c_polish = _local_optimize_penalty(c_polish, maxiter=150, ftol=1e-15, gtol=1e-11)
            
            score_polish = _score(c_polish)
            if score_polish > best_score:
                best_score = score_polish
                best_c = c_polish.copy()
    
    # Phase 3: Random perturbations and re-optimize
    if time.time() - start_time < deadline - 5.0:
        for _ in range(8):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Perturb
            c_pert = best_c + 0.005 * np.random.randn(N, 3)
            c_pert = np.clip(c_pert, -0.495, 0.495)
            
            # Quick smooth optimization
            c_new = _local_optimize_smooth(c_pert, alpha=1000, maxiter=100, ftol=1e-13, gtol=1e-9)
            
            # Penalty polish
            c_new = _local_optimize_penalty(c_new, maxiter=80, ftol=1e-15, gtol=1e-11)
            
            score_new = _score(c_new)
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

# EVOLVE-BLOCK-END
