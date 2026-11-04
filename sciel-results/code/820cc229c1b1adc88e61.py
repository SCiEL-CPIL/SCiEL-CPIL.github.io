"""
Optimization of 100 equal spheres in a unit cube.
Uses a barrier method with L-BFGS-B optimization to maximize the minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_obj_and_grad(x, alpha):
    """
    Smooth approximation of maximizing the minimum of pairwise distances/2 and wall distances.
    We minimize -soft_min(all_distances, alpha) where all_distances includes
    pairwise distances (divided by 2) and wall distances.
    
    soft_min(a, alpha) = -log(sum(exp(-a_i / alpha))) * alpha
    
    Gradient w.r.t. centers is computed analytically.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Collect all "distance" values: pairwise/2 and wall distances
    # For soft min, we work with negative log-sum-exp of (-d_i / alpha)
    # Let's build the list of distances and their gradients
    
    # Wall distances
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_abs  # shape (n,)
    
    # Pairwise distances / 2
    # To avoid O(n^2) memory for n=100, we use upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = centers[i_idx] - centers[j_idx]
    pair_dists = np.sqrt(np.sum(diffs**2, axis=1) + 1e-15)  # shape (n*(n-1)/2,)
    pair_dists_half = pair_dists / 2.0
    
    # All distances
    all_dists = np.concatenate([pair_dists_half, wall_dists])
    m = len(all_dists)
    
    # Soft min: f = -alpha * log(sum(exp(-d_i / alpha)))
    # To avoid overflow, use the max trick
    neg_d_over_a = -all_dists / alpha
    max_neg = np.max(neg_d_over_a)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_d_over_a - max_neg)))
    f = -alpha * log_sum
    
    # Gradient of f w.r.t. each distance d_i:
    # df/dd_i = -alpha * (1/sum_exp) * (-1/alpha) * exp(-d_i/alpha)
    #          = exp(-d_i/alpha) / sum_exp
    exp_vals = np.exp(neg_d_over_a - max_neg)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (m,), sums to 1
    
    # Now compute gradient w.r.t. centers
    grad = np.zeros_like(centers)
    
    # Pairwise part: indices 0 to len(pair_dists_half)-1
    n_pairs = len(pair_dists_half)
    pair_weights = weights[:n_pairs]
    
    # For each pair (i, j), contribution to grad[i] and grad[j]:
    # d(pair_dist/2)/d(c_i) = (c_i - c_j) / (2 * pair_dist) = (c_i - c_j) / (2 * 2*pair_dist_half)
    # Actually: d(pair_dist)/d(c_i) = (c_i - c_j) / pair_dist
    # So d(pair_dist/2)/d(c_i) = (c_i - c_j) / (2 * pair_dist)
    # And df/d(c_i) = sum over pairs involving i of weight_k * d(dist_k)/d(c_i)
    
    # Vectorized: for each pair k with weight w_k:
    # grad[i_k] += w_k * (c_{i_k} - c_{j_k}) / (2 * pair_dist_k)
    # grad[j_k] -= w_k * (c_{i_k} - c_{j_k}) / (2 * pair_dist_k)
    
    # Factor: w_k / (2 * pair_dist_k) = w_k / pair_dists[k]
    factors = pair_weights / pair_dists  # shape (n_pairs,)
    
    # Accumulate using np.add.at for repeated indices
    # grad[i_idx] += factors[:, None] * diffs
    # grad[j_idx] -= factors[:, None] * diffs
    np.add.at(grad, i_idx, factors[:, np.newaxis] * diffs)
    np.add.at(grad, j_idx, -factors[:, np.newaxis] * diffs)
    
    # Wall part: indices n_pairs to m-1
    wall_weights = weights[n_pairs:]
    
    # d(wall_dist_i)/d(centers[i]) = -sign(centers[i,k]) for the coordinate k that achieves max_abs
    # So df/d(centers[i]) = wall_weights[i] * (-sign(...))
    # i.e., grad[i] -= wall_weights[i] * sign_vector_i
    
    for i in range(n):
        if wall_weights[i] < 1e-15:
            continue
        # Find which coordinate achieves the max
        max_val = max_abs[i]
        # Use the first coordinate that is close to max
        for k in range(3):
            if np.isclose(abs_c[i, k], max_val, atol=1e-10):
                grad[i, k] -= wall_weights[i] * np.sign(centers[i, k])
                break
    
    return f, grad.flatten()

def _optimize_soft_min(init_centers, alpha, max_iter=500, time_budget=8.0):
    """
    Optimize using soft-min objective with L-BFGS-B.
    """
    def obj_and_grad(x):
        val, g = _soft_min_obj_and_grad(x, alpha)
        return val, g
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return init_centers

def _make_grid(nx, ny, nz):
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append([x, y, z])
    return np.array(grid)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Grid-based inits with different orientations
    inits.append(_make_grid(5, 4, 5))
    inits.append(_make_grid(4, 5, 5))
    inits.append(_make_grid(5, 5, 4))
    
    # Body-centered cubic style: 4x4x6 = 96, add 4 more
    # Actually let's just use random inits with various seeds
    for seed in range(10):
        r = np.random.RandomState(seed)
        inits.append(r.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Anneal alpha from large to small for each init
    # Large alpha = smooth, easy to optimize; small alpha = closer to true min
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        
        current = init.copy()
        
        # Multi-scale alpha annealing
        alphas = [0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005]
        
        for alpha in alphas:
            if time.time() > deadline - 15.0:
                break
            
            time_budget = min(10.0, max(2.0, deadline - 10.0 - time.time()))
            if time_budget < 2.0:
                break
            
            optimized = _optimize_soft_min(current, alpha, max_iter=300, time_budget=time_budget)
            
            # Check improvement
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
            
            # Only continue if we made progress or are close
            current = optimized
            
            # Early break if we're at a very good radius and alpha is small
            if best_radius > 0.111 and alpha < 0.01:
                break
    
    # Final polishing with very small alpha on the best config
    if best_centers is not None and time.time() < deadline - 10.0:
        for alpha in [0.003, 0.002, 0.001]:
            if time.time() > deadline - 5.0:
                break
            polished = _optimize_soft_min(best_centers, alpha, max_iter=500, time_budget=8.0)
            r_pol = _compute_radius(polished)
            if r_pol > best_radius:
                best_radius = r_pol
                best_centers = polished.copy()
    
    # Safety: ensure valid output
    if best_centers is None:
        best_centers = inits[0]
    else:
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
