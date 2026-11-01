import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def _compute_scores(c):
    """
    Compute the true radius and check validity.
    c: (N, 2) array
    """
    if not np.all(np.isfinite(c)):
        return 0.0, False
    
    # Check bounds
    if np.any(np.abs(c) > 0.5):
        return 0.0, False
        
    # Pairwise distances
    # Using broadcasting for vectorization
    # diff[i, j, k] = c[i, k] - c[j, k]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist = np.min(dists[mask])
    
    # Boundary distances
    # min(0.5 - x, 0.5 + x) for each coordinate
    boundary_dists = np.min(0.5 - np.abs(c), axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if r < 0:
        return 0.0, False
        
    return r, True

def _soft_min_obj_and_grad(c_flat, beta):
    """
    Objective: -S(c) where S(c) is the soft-min approximation of the min distance.
    c_flat: (2*N,) array
    """
    c = c_flat.reshape(N, 2)
    
    # 1. Compute distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero in gradient calculation if dist is 0
    # Add small epsilon
    eps = 1e-12
    dists_safe = np.maximum(dists, eps)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Pairwise part
    # U = -1/beta * log(sum(exp(-beta * d_ij)))
    # To avoid overflow/underflow, subtract max exponent
    exponents_p = -beta * dists
    exponents_p[~mask] = -np.inf # Ignore diagonal
    
    max_exp_p = np.max(exponents_p)
    if np.isinf(max_exp_p):
        return 1e10, np.zeros_like(c_flat)
        
    sum_exp_p = np.sum(np.exp(exponents_p - max_exp_p))
    if sum_exp_p == 0:
        return 1e10, np.zeros_like(c_flat)
        
    U = -1.0/beta * (np.log(sum_exp_p) + max_exp_p)
    
    # Weights for pairwise
    # w_ij = exp(-beta * d_ij) / sum_exp
    # Note: exp(exponents_p - max_exp_p) / sum_exp_p is the normalized weight
    weights_p = np.exp(exponents_p - max_exp_p) / sum_exp_p
    
    # Boundary part
    # b_ik = 0.5 - |c_ik|
    b_ik = 0.5 - np.abs(c)
    
    exponents_b = -beta * b_ik
    max_exp_b = np.max(exponents_b)
    if np.isinf(max_exp_b):
        return 1e10, np.zeros_like(c_flat)
        
    sum_exp_b = np.sum(np.exp(exponents_b - max_exp_b))
    if sum_exp_b == 0:
        return 1e10, np.zeros_like(c_flat)
        
    V = -1.0/beta * (np.log(sum_exp_b) + max_exp_b)
    
    weights_b = np.exp(exponents_b - max_exp_b) / sum_exp_b
    
    # Combine U and V into S
    # S = -1/(2*beta) * log(exp(-beta*U) + exp(-beta*V))
    exp_U = np.exp(-beta * U)
    exp_V = np.exp(-beta * V)
    
    denom = exp_U + exp_V
    if denom == 0:
        return 1e10, np.zeros_like(c_flat)
        
    S = -1.0/(2.0*beta) * np.log(denom)
    
    # Gradient
    # dS/dc = 0.5 * (exp_U * dU/dc + exp_V * dV/dc) / denom
    
    # dU/dc = sum_j w_ij * dd_ij/dc
    # dV/dc = sum_k w_ik * db_ik/dc
    
    grad_U = np.zeros_like(c)
    grad_V = np.zeros_like(c)
    
    # Compute dd/dc
    # dd_ij/dc_i = (c_i - c_j) / d_ij
    # dd_ij/dc_j = (c_j - c_i) / d_ij
    
    # Vectorized gradient for pairwise distances
    # We need to sum over j for each i
    # diff[i, j] = c_i - c_j
    # grad_dist[i, j] = diff[i, j] / dists[i, j]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_dists = diff / dists_safe[:, :, np.newaxis]
        
    # Zero out diagonal
    grad_dists[mask] = grad_dists[mask] # This is tricky with broadcasting, let's do it properly
    # Actually, diff[i,i] is 0, so grad is 0/0. We set dists_safe to eps, so 0/eps = 0.
    # But we must ensure diagonal weights are 0.
    # weights_p has 0 on diagonal because exp(-inf) is 0.
    
    # Sum weighted gradients
    # grad_U[i] = sum_j weights_p[i, j] * grad_dists[i, j]
    grad_U = np.sum(weights_p[:, :, np.newaxis] * grad_dists, axis=1)
    
    # dV/dc
    # db_ik/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    # Handle c=0, sign is 0, gradient is 0 (kink, but subgradient 0 is fine for L-BFGS usually)
    grad_b = -sign_c
    
    # grad_V[i] = sum_k weights_b[i, k] * grad_b[i, k]
    grad_V = np.sum(weights_b[:, :, np.newaxis] * grad_b, axis=1)
    
    # Combine
    grad_S = 0.5 * (exp_U * grad_U + exp_V * grad_V) / denom
    
    # We want to minimize -S
    obj = -S
    grad = -grad_S.flatten()
    
    return obj, grad

def _construct_grid():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _construct_hex():
    # Hexagonal packing
    # Radius r, side length 2r
    # Horizontal spacing: 2r
    # Vertical spacing: r * sqrt(3)
    # We don't know r yet, so we just generate a lattice and scale later?
    # Or generate a dense lattice and pick 50 points.
    # Let's generate points in a larger area and clip/scale.
    pts = []
    # Hex grid
    for row in range(10):
        for col in range(10):
            x = col * 1.0
            y = row * 0.866
            if row % 2 == 1:
                x += 0.5
            pts.append([x, y])
    pts = np.array(pts)
    # Normalize to fit in [-0.5, 0.5]
    # Find bounds
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    scale = 1.0 / max(width, height)
    pts = (pts - (min_x, min_y)) * scale - 0.5
    # Shuffle and take 50? Or just take first 50?
    # Taking first 50 from a sorted list might be biased.
    # Let's randomize the selection from the hex grid points that fall in the square.
    # Actually, just generate a specific hex pattern that fits well.
    # For N=50, a 7x7 grid is 49. A hex grid is denser.
    # Let's just use a random perturbation of the grid as a robust initializer.
    return _construct_grid() + np.random.normal(0, 0.01, (N, 2))

def _construct_random():
    return np.random.uniform(-0.5, 0.5, (N, 2))

def construct():
    best_c = _construct_grid()
    best_r, _ = _compute_scores(best_c)
    
    initializers = [
        _construct_grid(),
        _construct_hex(),
        _construct_random(),
        _construct_random()
    ]
    
    # Beta schedule
    betas = [50, 100, 200, 500, 1000, 2000]
    
    for init_c in initializers:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        current_c = init_c.copy()
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            # Flatten c
            c_flat = current_c.flatten()
            
            try:
                # Run L-BFGS-B
                # We don't need bounds because the objective penalizes going out,
                # but we can add bounds to help.
                # Bounds: [-0.5, 0.5] for each coordinate
                bounds = [(-0.5, 0.5)] * (2*N)
                
                res = minimize(
                    _soft_min_obj_and_grad,
                    c_flat,
                    method='L-BFGS-B',
                    jac=True,
                    args=(beta,),
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                new_c = res.x.reshape(N, 2)
                # Check if new_c is valid and better
                new_r, valid = _compute_scores(new_c)
                
                if valid and new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
                    
                # Use the result as the start for the next beta
                # Even if it didn't improve the global best, it might be a better local start
                current_c = new_c
                
            except Exception as e:
                # If optimization fails, keep current_c
                pass
                
    # Final polish with very high beta
    if time.time() - START_TIME < TIME_LIMIT - 2:
        c_flat = best_c.flatten()
        bounds = [(-0.5, 0.5)] * (2*N)
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                c_flat,
                method='L-BFGS-B',
                jac=True,
                args=(5000,),
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_c = res.x.reshape(N, 2)
            new_r, valid = _compute_scores(new_c)
            if valid and new_r > best_r:
                best_r = new_r
                best_c = new_c
        except:
            pass
            
    return best_c

if __name__ == "__main__":
    c = construct()
    r, valid = _compute_scores(c)
    print(f"Radius: {r}, Valid: {valid}")
