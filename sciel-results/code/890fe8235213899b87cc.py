import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0 # Safety margin

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _energy_and_grad(centers, target_d, alpha=10.0):
    """
    Energy function:
    1. Repulsion term: sum 1/d^alpha to keep spheres apart.
    2. Target violation: sum max(0, target_d - d)^2 to push min distance up.
    3. Wall penalty: sum max(0, 0.5 - |c|)^2 to keep spheres inside.
    """
    c = centers.reshape(-1, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0) # Avoid div by zero
    
    # 1. Repulsion Energy
    # Use a high power to strongly penalize close pairs
    # To avoid overflow/underflow, we can clip distances
    safe_dists = np.where(dists < 1e-6, 1e-6, dists)
    repulsion = np.sum(1.0 / (safe_dists**alpha))
    
    # Gradient of repulsion:
    # d/dx_i (1/d^alpha) = -alpha * d^(-alpha-1) * d/dx_i d
    # d/dx_i d = (x_i - x_j) / d
    # grad_i = sum_j -alpha * d^(-alpha-1) * (x_i - x_j)/d = -alpha * d^(-alpha-2) * (x_i - x_j)
    # Note: dists is symmetric.
    # grad_repulsion_i = -alpha * sum_{j!=i} (x_i - x_j) / d_{ij}^{alpha+2}
    
    # We compute this efficiently
    # factor[i,j] = alpha / d_{ij}^{alpha+2}
    # Only for i != j
    factors = np.zeros_like(dists)
    mask = dists > 1e-6
    factors[mask] = alpha / (dists[mask] ** (alpha + 2))
    np.fill_diagonal(factors, 0.0)
    
    # grad_rep = - sum_j factors[i,j] * (c_i - c_j)
    # = - sum_j factors[i,j] * c_i + sum_j factors[i,j] * c_j
    # = - c_i * sum_j factors[i,j] + sum_j factors[i,j] * c_j
    
    sum_factors = np.sum(factors, axis=1, keepdims=True) # Shape (N, 1)
    weighted_sum_c = np.sum(factors[:, :, np.newaxis] * c[np.newaxis, :, :], axis=1) # Shape (N, 3)
    
    grad_rep = -c * sum_factors + weighted_sum_c
    
    # 2. Target Violation Energy
    # V = max(0, target_d - d)^2
    violations = np.maximum(0.0, target_d - dists)
    # Set diagonal to 0
    np.fill_diagonal(violations, 0.0)
    energy_target = np.sum(violations**2)
    
    # Gradient:
    # d/dx_i (target_d - d)^2 = 2(target_d - d) * (-d/dx_i d) = -2(target_d - d) * (x_i - x_j)/d
    # Only if target_d > d
    active = dists < target_d
    # grad_target_i = - sum_j 2 * (target_d - d_{ij}) * (c_i - c_j) / d_{ij}  (for active)
    
    # weights[i,j] = 2 * violations[i,j] / dists[i,j]
    safe_dists_t = np.where(dists > 1e-6, dists, 1.0)
    weights_t = np.zeros_like(dists)
    weights_t[active] = 2.0 * violations[active] / safe_dists_t[active]
    np.fill_diagonal(weights_t, 0.0)
    
    # grad_target = - sum_j weights_t[i,j] * (c_i - c_j)
    # = - c_i * sum_j weights_t[i,j] + sum_j weights_t[i,j] * c_j
    
    sum_weights_t = np.sum(weights_t, axis=1, keepdims=True)
    weighted_sum_c_t = np.sum(weights_t[:, :, np.newaxis] * c[np.newaxis, :, :], axis=1)
    
    grad_target = -c * sum_weights_t + weighted_sum_c_t
    
    # 3. Wall Penalty
    # P = sum max(0, 0.5 - |c|)^2
    # For each coordinate k:
    # if c_ik > 0: P_k = max(0, 0.5 - c_ik)^2
    # if c_ik < 0: P_k = max(0, 0.5 + c_ik)^2
    # if c_ik = 0: 0
    
    abs_c = np.abs(c)
    wall_viol = np.maximum(0.0, 0.5 - abs_c)
    energy_wall = np.sum(wall_viol**2)
    
    # Gradient:
    # d/dc_ik max(0, 0.5 - |c_ik|)^2
    # If c_ik > 0: 2(0.5 - c_ik)(-1) = -2(0.5 - c_ik)
    # If c_ik < 0: 2(0.5 + c_ik)(1) = 2(0.5 + c_ik)
    # If c_ik = 0: 0 (subgradient, but 0 is fine for optimization)
    
    # sign(c) is +1 for c>0, -1 for c<0.
    # grad = -2 * wall_viol * sign(c)
    signs = np.sign(c)
    grad_wall = -2.0 * wall_viol * signs
    
    total_energy = repulsion + energy_target + energy_wall
    total_grad = (grad_rep + grad_target + grad_wall).flatten()
    
    return total_energy, total_grad

def construct():
    t0 = time.time()
    
    # Initial grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Iterative optimization
    # Start with a target distance slightly above the current min distance
    current_target_d = 2.0 * best_r * 1.01
    
    # We will try to increase the target distance step by step
    # If optimization fails to reach the target, we might lower the target or try a different start
    
    alpha = 10.0 # Power for repulsion
    
    for step in range(50): # Max steps
        if time.time() - t0 > DEADLINE:
            break
            
        # Add small noise to break symmetry if stuck
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        def objective(x):
            e, g = _energy_and_grad(x, current_target_d, alpha)
            return e, g
            
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'gtol': 1e-6, 'disp': False}
            )
            opt_pts = result.x.reshape(-1, 3)
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5, 0.5)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
                    # Increase target distance for next iteration
                    current_target_d = 2.0 * best_r * 1.02
                else:
                    # If no improvement, maybe try a slightly lower target or different noise
                    # Or just break if we are stuck
                    if step > 5:
                        # Try a few random restarts if stuck
                        if time.time() - t0 < DEADLINE - 10:
                             for _ in range(5):
                                 if time.time() - t0 > DEADLINE: break
                                 rand_pts = rng.uniform(-0.4, 0.4, size=(N, 3))
                                 def obj_rand(x):
                                     e, g = _energy_and_grad(x, 2.0*best_r*1.01, alpha)
                                     return e, g
                                 res_rand = minimize(obj_rand, rand_pts.flatten(), method='L-BFGS-B', jac=True, 
                                                    bounds=[(-0.5, 0.5)] * (N * 3), options={'maxiter': 100})
                                 rp = res_rand.x.reshape(-1, 3)
                                 rp = np.clip(rp, -0.5, 0.5)
                                 r_rp = _score(rp)
                                 if r_rp > best_r:
                                     best_r = r_rp
                                     best_pts = rp.copy()
                                     current_target_d = 2.0 * best_r * 1.02
                        break
        except Exception:
            pass
            
    return best_pts
