"""Improved program: Multi-start smooth annealing with adaptive penalty polish for 100 Spheres."""
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
    
    # Numerically stable log-sum-exp
    max_term = np.max(-alpha * all_dists)
    exp_terms = np.exp(-alpha * all_dists - max_term)
    
    Z = np.sum(exp_terms)
    if Z == 0:
        return 0.0, np.zeros_like(c)
        
    obj_val = (1.0 / alpha) * (np.log(Z) + max_term)
    
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


def _compute_penalty_obj_grad(c, target_r):
    """
    Penalty-based objective for final polish.
    Minimizes sum of squared violations of target radius constraint.
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
    
    violations = np.maximum(0, target_r - all_dists)
    
    obj_val = np.sum(violations ** 2)
    
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    viol_pair = violations[:len(pair_dists)]
    active_pair = viol_pair > 1e-14
    if np.any(active_pair):
        pair_dists_raw = d[i_idx[active_pair], j_idx[active_pair]]
        pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
        
        ci = c[i_idx[active_pair]]
        cj = c[j_idx[active_pair]]
        diff = ci - cj
        norms = pair_dists_raw_safe[:, np.newaxis]
        u = diff / norms
        
        w = 2.0 * viol_pair[active_pair][:, np.newaxis]
        
        idx_i = i_idx[active_pair]
        idx_j = j_idx[active_pair]
        
        np.add.at(grad_c, idx_i, -w * u / 2.0)
        np.add.at(grad_c, idx_j, w * u / 2.0)
    
    # Wall part
    viol_wall = violations[len(pair_dists):]
    ww = viol_wall.reshape(6, N)
    
    grad_c[:, 0] += 2.0 * ww[0] - 2.0 * ww[1]
    grad_c[:, 1] += 2.0 * ww[2] - 2.0 * ww[3]
    grad_c[:, 2] += 2.0 * ww[4] - 2.0 * ww[5]
    
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


def _initial_bcc():
    """Body-centered cubic inspired initial configuration."""
    centres = []
    for layer in range(4):
        z = -0.375 + layer * 0.25
        for row in range(5):
            y = -0.4 + row * 0.2
            for col in range(5):
                x = -0.4 + col * 0.2
                if layer % 2 == 1:
                    x += 0.1
                    y += 0.1
                centres.append([x, y, z])
    return np.array(centres[:N])


def _initial_fcc():
    """Face-centered cubic inspired initial configuration."""
    centres = []
    for layer in range(5):
        z = -0.4 + layer * 0.2
        for row in range(5):
            y = -0.4 + row * 0.2
            for col in range(4):
                x = -0.375 + col * 0.25
                if layer % 2 == 1:
                    x += 0.125
                    y += 0.1
                centres.append([x, y, z])
    return np.array(centres[:N])


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
        offset_x = (0.1 if idx % 2 == 1 else 0.0)
        offset_y = (0.1 if idx % 2 == 1 else 0.0)
        for y in ys:
            for x in xs:
                centres.append([x + offset_x, y + offset_y, z])
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


def _initial_random(seed=42):
    """Random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))


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
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        
        if result.success or result.fun < obj(x0) - 1e-12:
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            return c_new
    except Exception:
        pass
    
    return c


def _local_optimize_penalty(c, target_r, maxiter=100, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization on the penalty objective for final polish."""
    def obj(x):
        c_flat = x.reshape(N, 3)
        val, _ = _compute_penalty_obj_grad(c_flat, target_r)
        return val
    
    def grad(x):
        c_flat = x.reshape(N, 3)
        _, g = _compute_penalty_obj_grad(c_flat, target_r)
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
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        
        if result.success or result.fun < obj(x0) - 1e-12:
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            return c_new
    except Exception:
        pass
    
    return c


def _critical_constraint_polish(c, max_rounds=10, time_limit=20.0):
    """
    Polish by iteratively increasing the target radius and solving
    the penalty problem to equalize binding constraints.
    """
    start_time = time.time()
    current_r = _score(c)
    c_best = c.copy()
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        # Try to increase the radius slightly
        target_r = current_r + 0.0003
        
        c_new = _local_optimize_penalty(c_best, target_r, maxiter=80, ftol=1e-15, gtol=1e-11)
        
        new_score = _score(c_new)
        if new_score > current_r:
            current_r = new_score
            c_best = c_new.copy()
        else:
            # Try smaller increment
            target_r = current_r + 0.0001
            c_new = _local_optimize_penalty(c_best, target_r, maxiter=60, ftol=1e-15, gtol=1e-11)
            new_score = _score(c_new)
            if new_score > current_r:
                current_r = new_score
                c_best = c_new.copy()
            else:
                break
    
    return c_best, current_r


def _anneal_from_start(c_init, alphas, start_time, deadline):
    """Run the annealing schedule from a given initial configuration."""
    c = c_init.copy()
    best_score = _score(c)
    best_c = c.copy()
    
    for alpha in alphas:
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Use tighter tolerances as alpha increases
        if alpha >= 1000:
            ftol = 1e-14
            gtol = 1e-10
            maxiter = 300
        elif alpha >= 300:
            ftol = 1e-13
            gtol = 1e-9
            maxiter = 250
        elif alpha >= 80:
            ftol = 1e-12
            gtol = 1e-8
            maxiter = 200
        else:
            ftol = 1e-12
            gtol = 1e-8
            maxiter = 150
        
        c_new = _local_optimize_smooth(c, alpha, maxiter=maxiter, ftol=ftol, gtol=gtol)
        
        score_new = _score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
            c = c_new.copy()
    
    return best_c, best_score


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    np.random.seed(42)
    
    # Try multiple initial configurations
    init_configs = [
        _initial_bcc(),
        _initial_fcc(),
        _initial_grid(),
        _initial_grid_v2(),
        _initial_grid_v3(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(456)
    ]
    
    # Find the best initial configuration by score
    best_init = None
    best_init_score = 0.0
    for init_c in init_configs:
        score = _score(init_c)
        if score > best_init_score:
            best_init_score = score
            best_init = init_c.copy()
    
    # Phase 1: Smooth log-sum-exp annealing
    alphas = [5, 10, 20, 40, 80, 160, 300, 500, 800, 1200, 1600, 2000, 2500]
    
    best_c, best_score = _anneal_from_start(best_init, alphas, start_time, deadline)
    
    # Phase 2: Critical constraint polish
    if time.time() - start_time < deadline - 15.0:
        c_polish, score_polish = _critical_constraint_polish(best_c, max_rounds=15, time_limit=15.0)
        
        if score_polish > best_score:
            best_score = score_polish
            best_c = c_polish.copy()
    
    # Phase 3: Random perturbations and re-optimize
    if time.time() - start_time < deadline - 8.0:
        for trial in range(15):
            if time.time() - start_time > deadline - 8.0:
                break
            
            # Perturb with decreasing magnitude
            pert_mag = 0.004 * (1.0 - trial / 15.0) + 0.001
            c_pert = best_c + pert_mag * np.random.randn(N, 3)
            c_pert = np.clip(c_pert, -0.495, 0.495)
            
            # Quick smooth optimization
            c_new = _local_optimize_smooth(c_pert, alpha=1500, maxiter=100, ftol=1e-13, gtol=1e-9)
            
            # Critical constraint polish
            c_new, score_new = _critical_constraint_polish(c_new, max_rounds=5, time_limit=5.0)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
    
    # Phase 4: Try a few different initial configurations if time permits
    if time.time() - start_time < deadline - 20.0:
        for init_c in init_configs[1:4]:
            if time.time() - start_time > deadline - 20.0:
                break
            
            c_try, score_try = _anneal_from_start(init_c, alphas[3:], start_time, deadline)
            if score_try > best_score:
                best_score = score_try
                best_c = c_try.copy()
            
            c_polish, score_polish = _critical_constraint_polish(best_c, max_rounds=10, time_limit=10.0)
            if score_polish > best_score:
                best_score = score_polish
                best_c = c_polish.copy()
    
    # Final polish with very tight tolerances
    if time.time() - start_time < deadline - 3.0:
        c_final, score_final = _critical_constraint_polish(best_c, max_rounds=8, time_limit=10.0)
        if score_final > best_score:
            best_score = score_final
            best_c = c_final.copy()

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

# EVOLVE-BLOCK-END
