import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.spatial.distance import pdist

N = 100

def _compute_smooth_obj_grad(x, beta=100.0):
    """
    Compute the smooth log-sum-exp objective and gradient.
    Objective: -soft_min(margins)
    Margins: pairwise distances / 2, and wall distances.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    # pdist is efficient
    pd = pdist(centers, metric='euclidean') / 2.0
    
    # Wall distances
    # Distance to wall is 0.5 - |c_ik|
    wall_dists = 0.5 - np.abs(centers)
    wall_flat = wall_dists.flatten()
    
    all_margins = np.concatenate([pd, wall_flat])
    
    # Soft min
    min_val = np.min(all_margins)
    # Numerical stability
    # exp(-beta * (x - min_val))
    exp_vals = np.exp(-beta * (all_margins - min_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = min_val - log_sum / beta
    
    obj = -soft_min
    
    # Gradient
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    ii, jj = np.triu_indices(n, k=1)
    diffs = centers[ii] - centers[jj]
    dists = np.linalg.norm(diffs, axis=1)
    dists_safe = np.maximum(dists, 1e-12)
    
    # d(margin)/d(c_i) = (c_i - c_j) / (2 * dist)
    pair_grads = diffs / (2.0 * dists_safe[:, None])
    
    # Scatter weights
    # We need to sum weights[k] * pair_grads[k] for each i and j
    # Using bincount or loop? Loop is O(N^2) which is 10k, fine.
    # But vectorized is better.
    # grad[ii] += weights[:, None] * pair_grads
    # grad[jj] -= weights[:, None] * pair_grads
    
    # This is a scatter-add.
    # np.add.at is slow.
    # Let's use the loop, it's fast enough for N=100 (4950 pairs).
    for k in range(len(ii)):
        w = weights[k]
        grad[ii[k]] += w * pair_grads[k]
        grad[jj[k]] -= w * pair_grads[k]
        
    # Wall gradient
    # d(margin)/d(c_ik) = -sign(c_ik)
    wall_grads = -np.sign(centers)
    wall_weights = weights[len(pd):].reshape(-1, 3)
    grad += wall_weights * wall_grads
    
    # Objective gradient is -grad
    return obj, -grad

def _score(centers):
    """Compute the exact radius."""
    centers = centers.reshape(-1, 3)
    # Wall
    wall_dists = 0.5 - np.abs(centers)
    wall_min = np.min(wall_dists)
    
    # Pair
    pd = pdist(centers, metric='euclidean') / 2.0
    pair_min = np.min(pd)
    
    return min(wall_min, pair_min)

def _initial_jittered_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Jitter
    np.random.seed(42)
    jitter = np.random.uniform(-0.01, 0.01, grid.shape)
    return grid[:N] + jitter

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = _initial_jittered_grid()
    best_radius = _score(best_centers)
    
    # 1. Differential Evolution Phase
    # Use a smooth objective with moderate beta
    def de_obj(x):
        if time.time() > deadline - 10.0:
            return 1e9
        return _compute_smooth_obj_grad(x, beta=50.0)[0]
        
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    try:
        # Run DE
        # maxiter=100, popsize=20 -> 2000 evals
        # This should take ~1-2 seconds
        res_de = differential_evolution(
            de_obj, bounds, 
            maxiter=100, popsize=20, 
            seed=42, tol=1e-8, 
            polish=False,
            init='sobol' # Better initialization
        )
        de_centers = res_de.x.reshape(-1, 3)
        de_radius = _score(de_centers)
        
        if de_radius > best_radius:
            best_radius = de_radius
            best_centers = de_centers
    except Exception as e:
        pass
        
    # 2. Local Polish Phase (L-BFGS-B)
    # Use the best candidate from DE or initial
    x0 = best_centers.flatten()
    
    # Increase beta for sharper minimum
    def polish_obj_grad(x):
        if time.time() > deadline - 5.0:
            return 1e9, np.zeros_like(x)
        return _compute_smooth_obj_grad(x, beta=200.0)
        
    try:
        res_polish = minimize(
            polish_obj_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        if res_polish.fun < -best_radius:
            polished = res_polish.x.reshape(-1, 3)
            # Clip to ensure validity
            polished = np.clip(polished, -0.4999, 0.4999)
            r_polished = _score(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
    except Exception:
        pass
        
    # 3. Shake and Polish Loop (if time remains)
    # Try to escape local minima by perturbing and re-polishing
    np.random.seed(123)
    while time.time() < deadline - 10.0:
        # Perturb
        noise = np.random.normal(0, 0.01, best_centers.shape)
        perturbed = best_centers + noise
        perturbed = np.clip(perturbed, -0.499, 0.499)
        
        x_pert = perturbed.flatten()
        
        try:
            res_shake = minimize(
                polish_obj_grad,
                x_pert,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-9}
            )
            if res_shake.fun < -best_radius:
                shaken = res_shake.x.reshape(-1, 3)
                shaken = np.clip(shaken, -0.4999, 0.4999)
                r_shaken = _score(shaken)
                if r_shaken > best_radius:
                    best_radius = r_shaken
                    best_centers = shaken
        except Exception:
            break
            
    return best_centers
