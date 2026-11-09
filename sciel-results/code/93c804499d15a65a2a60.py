import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(c):
    """Calculate the radius for a given configuration."""
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(np.abs(c) > 0.5 + 1e-9):
        return 0.0
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(c))
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _objective_and_grad(x_flat, lambda_val):
    x = x_flat.reshape(N, 3)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    abs_x = np.abs(x)
    W = 0.5 - abs_x
    
    upper_idx = np.triu_indices(N, k=1)
    pair_dists = dists[upper_idx]
    pair_terms = np.exp(-lambda_val * pair_dists)
    obj_pair = np.sum(pair_terms)
    
    wall_terms = np.exp(-lambda_val * W)
    obj_wall = np.sum(wall_terms)
    
    obj = obj_pair + obj_wall
    
    mask = ~np.eye(N, dtype=bool)
    coeff = np.zeros_like(dists)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    coeff[mask] = -lambda_val * np.exp(-lambda_val * dists[mask]) / safe_dists[mask]
    
    Sum_C = coeff.sum(axis=1)
    Term1 = Sum_C[:, np.newaxis] * x
    Term2 = coeff @ x
    grad_pair = Term1 - Term2
    
    sign_x = np.sign(x)
    grad_wall = lambda_val * sign_x * wall_terms
    
    grad = grad_pair + grad_wall
    
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_radius = 0.0
    best_config = None
    
    configs = []
    
    # Grid 4x4x4
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Init 1: Random subset of grid
    idx1 = np.random.choice(64, 50, replace=False)
    configs.append(grid[idx1].copy())
    
    # Init 2: Perturbed grid subset
    idx2 = np.random.choice(64, 50, replace=False)
    c2 = grid[idx2].copy()
    c2 += np.random.normal(0, 0.1, c2.shape)
    np.clip(c2, -0.5, 0.5, out=c2)
    configs.append(c2)
    
    # Init 3: Random
    configs.append(np.random.uniform(-0.5, 0.5, (N, 3)))
    
    # Init 4: Another random subset of grid
    idx3 = np.random.choice(64, 50, replace=False)
    configs.append(grid[idx3].copy())

    lambdas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    
    for i, init_config in enumerate(configs):
        if time.time() > deadline:
            break
            
        current_config = init_config.copy()
        
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            def obj_func(x_flat):
                obj, _ = _objective_and_grad(x_flat, lam)
                return obj
            
            def grad_func(x_flat):
                _, grad = _objective_and_grad(x_flat, lam)
                return grad
                
            try:
                res = minimize(
                    obj_func, 
                    current_config.flatten(), 
                    jac=grad_func, 
                    method='L-BFGS-B', 
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                current_config = res.x.reshape(N, 3)
            except Exception:
                pass
        
        r = _score(current_config)
        if r > best_radius:
            best_radius = r
            best_config = current_config.copy()
            
    if best_config is None:
        best_config = configs[0]
        
    return best_config
