"""Combined program: Smooth Log-Sum-Exp Objective with Annealing for 100 Spheres."""
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
    
    Objective: f(c) = (1/alpha) * log( sum_{s in S} exp(-alpha * s) )
    where S contains:
    - d_ij / 2 for all i < j
    - 0.5 - |c_ik| for all i, k
    
    We want to minimize f(c), which maximizes the smooth approximation of the radius.
    """
    # Pairwise distances
    d = cdist(c, c)
    # We only need i < j to avoid double counting and self-distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0  # (N*(N-1)/2,)
    
    # Wall distances
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])  # (3*N,)
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Numerical stability: subtract max to avoid underflow?
    # exp(-alpha * s). If s is large, exp is 0. If s is small, exp is large.
    # The min s dominates.
    # Let's just compute directly. alpha is not too huge.
    exp_terms = np.exp(-alpha * all_dists)
    
    Z = np.sum(exp_terms)
    if Z == 0:
        # Fallback, shouldn't happen unless distances are inf
        return 0.0, np.zeros_like(c)
        
    obj_val = (1.0 / alpha) * np.log(Z)
    
    # Gradient
    # grad f = -1/Z * sum( exp(-alpha * s) * grad s )
    # grad s for pair (i,j):
    # s = ||c_i - c_j|| / 2
    # grad s / grad c_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # grad s / grad c_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # We need to accumulate gradients for each coordinate of each center.
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    exp_pair = exp_terms[:len(pair_dists)]
    weights_pair = exp_pair / Z  # (N*(N-1)/2,)
    
    # dists for pairs
    pair_dists_raw = d[i_idx, j_idx]
    # Avoid division by zero
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    # Vectorized gradient accumulation
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj  # (4950, 3)
    norms = pair_dists_raw_safe[:, np.newaxis]  # (4950, 1)
    
    # Unit vector from j to i
    u = diff / norms  # (4950, 3)
    
    # Weight for each pair
    # grad s = u / 2 for c_i, -u / 2 for c_j
    # So contribution to grad_c is -weights * grad_s
    # grad_c[i] -= weights * u / 2
    # grad_c[j] += weights * u / 2
    
    w = weights_pair[:, np.newaxis]  # (4950, 1)
    
    contrib = w * u / 2.0  # (4950, 3)
    
    # Accumulate
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall part
    exp_wall = exp_terms[len(pair_dists):]
    weights_wall = exp_wall / Z  # (3*N,)
    
    # Wall distances:
    # 0..N-1: 0.5 - c[:, 0] -> grad w.r.t c[:, 0] is -1
    # N..2N-1: 0.5 + c[:, 0] -> grad w.r.t c[:, 0] is 1
    # 2N..3N-1: 0.5 - c[:, 1] -> grad w.r.t c[:, 1] is -1
    # 3N..4N-1: 0.5 + c[:, 1] -> grad w.r.t c[:, 1] is 1
    # 4N..5N-1: 0.5 - c[:, 2] -> grad w.r.t c[:, 2] is -1
    # 5N..6N-1: 0.5 + c[:, 2] -> grad w.r.t c[:, 2] is 1
    
    # Reshape weights_wall to (6, N)
    ww = weights_wall.reshape(6, N)
    
    # grad_c[:, 0] gets contribution from ww[0] (grad -1) and ww[1] (grad +1)
    # Contribution to grad_c[i, 0] = -1 * ww[0, i] + 1 * ww[1, i]
    # Remember: grad f = - sum( w_s * grad s )
    # So grad_c[i, 0] -= ww[0, i] * (-1) + ww[1, i] * (1)
    #             = ww[0, i] - ww[1, i]
    
    grad_c[:, 0] += ww[0] - ww[1]
    grad_c[:, 1] += ww[2] - ww[3]
    grad_c[:, 2] += ww[4] - ww[5]
    
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
    xs = np.linspace(-0.375, 0.375, 4)  # spacing 0.25
    ys = np.linspace(-0.4, 0.4, 5)  # spacing 0.2
    zs = np.linspace(-0.4, 0.4, 5)  # spacing 0.2
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _local_optimize(c, alpha, maxiter=200):
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
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
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
    
    # Annealing schedule for alpha
    # Start small (smooth) to large (sharp)
    alphas = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    current_alpha_idx = 0
    c = best_c.copy()
    
    while time.time() - start_time < deadline - 10.0:
        # If we've gone through all alphas, increase the max or break
        if current_alpha_idx >= len(alphas):
            alpha = alphas[-1]
        else:
            alpha = alphas[current_alpha_idx]
            current_alpha_idx += 1
            
        # Optimize with current alpha
        c_new = _local_optimize(c, alpha, maxiter=200)
        
        score_new = _score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
            c = c_new.copy()
    
    # Final Polish:
    # Try a few random perturbations from the best found and re-optimize with high alpha
    if time.time() - start_time < deadline - 5.0:
        for _ in range(5):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Perturb
            c_pert = best_c + 0.01 * np.random.randn(N, 3)
            c_pert = np.clip(c_pert, -0.49, 0.49)
            
            alpha = 1000
            c_new = _local_optimize(c_pert, alpha, maxiter=100)
            
            score_new = _score(c_new)
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

# EVOLVE-BLOCK-END
