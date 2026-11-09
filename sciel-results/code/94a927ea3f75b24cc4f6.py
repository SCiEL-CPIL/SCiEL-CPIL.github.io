"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Distance to boundary
    d_boundary = np.min(0.5 - np.abs(centres), axis=1)
    min_boundary = np.min(d_boundary)
    # Pairwise distances
    # Use vectorized computation for speed
    # c_i - c_j for all pairs
    # To avoid O(N^2) memory blowup for large N (N=100 is small, so it's fine)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_boundary)


def _soft_min_radius_grad(centres, beta=50.0):
    """
    Compute soft-min radius and its gradient.
    f = -log(sum(exp(-beta * v))) / beta
    """
    n = len(centres)
    if n == 0:
        return 0.0, np.zeros_like(centres)
    
    vals = []
    grads = []
    
    # Boundary constraints
    for i in range(n):
        for k in range(3):
            v = 0.5 - abs(centres[i, k])
            vals.append(v)
            g = np.zeros(3 * n)
            if centres[i, k] >= 0:
                g[i * 3 + k] = -1.0
            else:
                g[i * 3 + k] = 1.0
            grads.append(g)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            diff = centres[i] - centres[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                v = 0.0
                g = np.zeros(3 * n)
            else:
                v = d / 2.0
                g = np.zeros(3 * n)
                g[i * 3:i * 3 + 3] = diff / (2.0 * d)
                g[j * 3:j * 3 + 3] = -diff / (2.0 * d)
            vals.append(v)
            grads.append(g)
    
    vals = np.array(vals)
    grads = np.array(grads)  # shape (m, 3n)
    
    # Soft-min: f = -log(sum(exp(-beta * v))) / beta
    shifted = -beta * vals
    max_shift = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shift)
    log_sum = max_shift + np.log(np.sum(exp_shifted))
    soft_min_val = -log_sum / beta
    
    # Gradient: df/dx = -sum_j (p_j * dv_j/dx) where p_j = exp(-beta*v_j) / sum_k exp(-beta*v_k)
    p = exp_shifted / np.sum(exp_shifted)
    grad = -np.sum(p[:, np.newaxis] * grads, axis=0)
    
    return soft_min_val, grad


def _init_grid():
    """
    Generate a jittered grid of 100 points.
    Uses a 5x5x5 grid, selects the 100 points with largest min-distance to boundary.
    """
    # 5x5x5 grid
    ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Calculate distance to boundary for each point
    d_boundary = np.min(0.5 - np.abs(points), axis=1)
    
    # Sort indices by distance to boundary (descending)
    idx = np.argsort(-d_boundary)
    
    # Take top 100
    selected_idx = idx[:N]
    points = points[selected_idx].copy()
    
    # Add small random jitter to break symmetry
    # Jitter scale should be small enough not to overlap, but large enough to help optimization
    # Max distance between neighbors is 0.2. Radius is ~0.1.
    # Jitter of 0.01 is safe.
    points += np.random.randn(N, 3) * 0.01
    
    # Clip to bounds
    points = np.clip(points, -0.49, 0.49)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration
    centres = _init_grid().astype(np.float64)
    
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Optimization parameters
    # We will run multiple rounds of L-BFGS-B with increasing beta
    # and restarts from perturbed best solutions.
    
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Try a few restarts for this beta
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            
            if restart == 0:
                x0 = best_centres.flatten()
            else:
                # Perturb the best solution found so far
                noise_scale = 0.02 * (1.0 + 0.5 * restart)
                x0 = (best_centres + np.random.randn(N, 3) * noise_scale).flatten()
                x0 = np.clip(x0, -0.49, 0.49)
            
            def objective(x):
                centres_flat = x.reshape(-1, 3)
                val, _ = _soft_min_radius_grad(centres_flat, beta)
                return -val
            
            def grad(x):
                centres_flat = x.reshape(-1, 3)
                _, g = _soft_min_radius_grad(centres_flat, beta)
                return g
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                result = minimize(
                    objective, x0, jac=grad, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                new_centres = result.x.reshape(-1, 3)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                r = _compute_radius(new_centres)
                
                if r > best_radius:
                    best_radius = r
                    best_centres = new_centres.copy()
                    
            except Exception:
                pass
    
    # Final polish with very high beta if time remains
    if time.time() < deadline - 5.0:
        beta_final = 500.0
        x0 = best_centres.flatten()
        
        def objective_final(x):
            centres_flat = x.reshape(-1, 3)
            val, _ = _soft_min_radius_grad(centres_flat, beta_final)
            return -val
        
        def grad_final(x):
            centres_flat = x.reshape(-1, 3)
            _, g = _soft_min_radius_grad(centres_flat, beta_final)
            return g
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result_final = minimize(
                objective_final, x0, jac=grad_final, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
            )
            
            new_centres = result_final.x.reshape(-1, 3)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            r = _compute_radius(new_centres)
            
            if r > best_radius:
                best_radius = r
                best_centres = new_centres.copy()
                
        except Exception:
            pass

    return best_centres
# EVOLVE-BLOCK-END
