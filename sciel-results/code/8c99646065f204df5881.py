import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _objective_and_grad(x_flat):
    """
    Calculate the logarithmic energy and its Riemannian gradient (projected to tangent space)
    for points on the unit sphere.
    x_flat: (3*N,) array
    Returns: energy (float), gradient (3*N,) array
    """
    x = x_flat.reshape(N, 3)
    
    # Normalize points to ensure they are on the sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms
    
    # Pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask for i < j to compute energy
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    # Using np.sum for speed in optimizer loop
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient
    # Euclidean gradient of E w.r.t x_i is sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    weights = 1.0 / (dist_sq + 1e-30)
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    # Project to tangent space
    g_dot_x = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tangent = grad_euclid - g_dot_x * x
    
    return energy, grad_tangent.flatten()

def _optimize_scipy(x0, time_limit=20.0):
    """
    Optimize using L-BFGS-B.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': 2000,
            'ftol': 1e-18,
            'gtol': 1e-12,
            'maxls': 50,
            'maxcor': 20,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    
    # Normalize again to ensure validity
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _energy_eval(x):
    """
    High precision energy calculation for final evaluation.
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    return -math.fsum(np.log(d_pairs).tolist())

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Perturbed Fibonacci Spirals
    inits = [x_fib]
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # 3. Random Initializations
    for _ in range(10):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
        
    # Optimize each initialization
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        try:
            time_left = deadline - time.time() - 10.0
            if time_left < 5.0:
                break
            t_alloc = min(10.0, time_left)
            
            x_opt, e_opt = _optimize_scipy(x0, time_limit=t_alloc)
            
            # Verify with high precision energy
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt
        except Exception:
            continue
            
    if best_x is None:
        best_x = x_fib
        best_e = _energy_eval(best_x)
        
    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        try:
            t_final = deadline - time.time() - 5.0
            x_final, e_final = _optimize_scipy(best_x, time_limit=t_final)
            e_final_precise = _energy_eval(x_final)
            if e_final_precise < best_e:
                best_x = x_final
                best_e = e_final_precise
        except Exception:
            pass
            
    # Ensure valid output
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    
    return best_x
