"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _time_left(deadline, margin=5.0):
    return deadline - time.time() - margin > 0


def _compute_radius(centres):
    """Compute the largest common radius for a given set of centres."""
    # Boundary distance
    boundary = np.minimum(0.5 - np.abs(centres), axis=1).min()
    # Pairwise distances
    from scipy.spatial.distance import pdist
    if len(centres) < 2:
        return boundary
    dists = pdist(centres, 'euclidean')
    min_pair = dists.min() / 2.0 if len(dists) > 0 else np.inf
    return min(boundary, min_pair)


def _soft_min_neg(x, beta):
    """Soft minimum of negative values (i.e., soft max of negative distances)."""
    # x is array of negative values; we want soft min which is smooth max of -x
    # soft_min_neg(x, beta) = -soft_max(-x, beta) = -(1/beta)*log(sum(exp(-beta*x_i)))
    # But we want to minimize the maximum violation, so we maximize the minimum distance
    # Let's use: soft_max of distances = (1/beta)*log(sum(exp(beta*d_i)))
    # We want to maximize this
    return -(1.0 / beta) * np.log(np.sum(np.exp(-beta * x)))


def _objective_and_grad(centres_flat, beta):
    """Objective: negative soft-max of min distance and boundary distance.
    Returns (value, grad)."""
    centres = centres_flat.reshape(-1, 2)
    N_c = len(centres)
    
    # Boundary distances
    boundary_dists = np.minimum(0.5 - np.abs(centres), axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(centres, centres, 'euclidean')
    np.fill_diagonal(D, np.inf)
    pair_dists = D / 2.0
    # Flatten and remove inf
    pair_flat = pair_dists[~np.isinf(pair_dists)]
    
    # Combine all "radius" candidates: boundary and pairwise
    all_dists = np.concatenate([boundary_dists, pair_flat])
    
    # Soft max
    max_d = all_dists.max()
    exp_shifted = np.exp(beta * (all_dists - max_d))
    log_sum = np.log(np.sum(exp_shifted)) / beta + max_d
    
    # Objective is -log_sum (we want to maximize)
    value = -log_sum
    
    # Gradient
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient w.r.t. boundary distances
    grad = np.zeros_like(centres)
    # d(boundary_dist_i)/d(c_ij): 
    # boundary_dist_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # This is piecewise; use subgradient approximation
    x = centres[:, 0]
    y = centres[:, 1]
    # For each point, the active boundary
    for i in range(N_c):
        # w_i is weight for boundary_i in all_dists
        w_b = weights[i]
        # subgrad of min(0.5-|x|, 0.5-|y|)
        bx = 0.5 - abs(x[i])
        by = 0.5 - abs(y[i])
        if bx < by - 1e-10:
            # active: 0.5 - |x|
            ddx = -np.sign(x[i])
            ddy = 0.0
        elif by < bx - 1e-10:
            # active: 0.5 - |y|
            ddx = 0.0
            ddy = -np.sign(y[i])
        else:
            # tie
            ddx = -0.5 * np.sign(x[i])
            ddy = -0.5 * np.sign(y[i])
        grad[i, 0] += -beta * w_b * ddx  # d(value)/d(x_i) = -beta * w * d(boundary)/d(x_i)
        grad[i, 1] += -beta * w_b * ddy
    
    # Gradient w.r.t. pairwise distances
    # For each pair (i,j), d(pair_dist)/d(c_i) and d(c_j)
    # pair_dist = |c_i - c_j| / 2
    # d/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
    for i in range(N_c):
        for j in range(i + 1, N_c):
            diff = centres[i] - centres[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            # index in all_dists: boundary_dists has N_c elements, then pairs
            # pair (i,j) index: i*N_c + j but only upper triangle... 
            # Let me recompute: all_dists = [b_0,...,b_{N-1}, d_{01}, d_{02},...,d_{0,N-1}, d_{12},...]
            # Number of pairs before (i,j): sum_{k=0}^{i-1} (N_c - k) + (j - i - 1)
            idx = N_c
            for k in range(i):
                idx += (N_c - k)
            idx += (j - i - 1)
            w_p = weights[idx]
            unit = diff / dist
            grad[i, 0] += -beta * w_p * unit[0] / 2.0
            grad[i, 1] += -beta * w_p * unit[1] / 2.0
            grad[j, 0] += -beta * w_p * (-unit[0]) / 2.0
            grad[j, 1] += -beta * w_p * (-unit[1]) / 2.0
    
    return value, grad.flatten()


def _initial_hex_grid(n):
    """Generate a hexagonal-like initial grid of n points in [-0.5, 0.5]^2."""
    # Start with a square grid and then perturb slightly
    m = int(np.ceil(np.sqrt(n)))
    # Use a denser grid and select best n
    m_dense = int(np.ceil(np.sqrt(n * 1.2)))
    ticks = (np.arange(m_dense) + 0.5) / m_dense - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Shuffle and take first n
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(grid))[:n]
    return grid[indices].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Initial configuration
    centres = _initial_hex_grid(N)
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Phase 1: Gradient-based optimization with increasing beta
    beta_start = 50.0
    beta_end = 500.0
    n_phases = 8
    
    for phase in range(n_phases):
        if not _time_left(deadline):
            break
        
        beta = beta_start + (beta_end - beta_start) * (phase / max(1, n_phases - 1))
        flat = centres.flatten().copy()
        
        # Run L-BFGS-B
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                flat,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centres = result.x.reshape(-1, 2)
            
            # Check if improved
            r = _compute_radius(new_centres)
            if r > best_radius:
                best_radius = r
                best_centres = new_centres.copy()
                centres = new_centres.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with coordinate ascent
    for iteration in range(50):
        if not _time_left(deadline):
            break
        
        improved = False
        for i in range(N):
            if not _time_left(deadline):
                break
            
            # Try small perturbations in each direction
            for dx in [-0.001, 0.001]:
                new_c = best_centres.copy()
                new_c[i, 0] += dx
                r = _compute_radius(new_c)
                if r > best_radius:
                    best_radius = r
                    best_centres = new_c
                    improved = True
            for dy in [-0.001, 0.001]:
                new_c = best_centres.copy()
                new_c[i, 1] += dy
                r = _compute_radius(new_c)
                if r > best_radius:
                    best_radius = r
                    best_centres = new_c
                    improved = True
        
        if not improved:
            break
    
    # Ensure validity: clamp to square
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END

