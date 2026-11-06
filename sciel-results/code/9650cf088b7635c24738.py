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
    
    # Pairwise differences using matrix operations for speed
    # dist_sq[i,j] = ||x_i - x_j||^2 = 2 - 2*x_i·x_j
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0) on diagonal
    
    # Mask for i < j to compute energy
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask])
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient computation
    # weights[i,j] = 1/||x_i - x_j||^2 for i != j
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    # Project to tangent space
    g_dot_x = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tangent = grad_euclid - g_dot_x * x
    
    return energy, grad_tangent.flatten()

def _optimize_l_bfgs(x0, time_limit=15.0):
    """
    Optimize using L-BFGS-B with moderate tolerances.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': 1500,
            'ftol': 1e-16,
            'gtol': 1e-10,
            'maxls': 40,
            'maxcor': 20,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _optimize_bfgs_tight(x0, time_limit=25.0):
    """
    High-precision polish using BFGS with very tight tolerances.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='BFGS', 
        jac=True,
        options={
            'maxiter': 800,
            'gtol': 1e-14,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _polish_newton(x0, time_limit=20.0):
    """
    Custom Newton-like polish: use BFGS Hessian approximation for final steps.
    We run L-BFGS-B with extremely tight tolerances as a proxy for Newton.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': 500,
            'ftol': 1e-18,
            'gtol': 1e-15,
            'maxls': 30,
            'maxcor': 20,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _energy_eval(x):
    """
    High precision energy calculation for final evaluation.
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask])
    return -math.fsum(np.log(d_pairs).tolist())

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Perturbed Fibonacci Spirals (fewer, larger scales)
    for scale in [0.05, 0.2, 0.8]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # 3. Random Initializations (fewer)
    for _ in range(5):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
    
    # 4. Antipodal pairs + jitter
    for _ in range(2):
        x_base = rng.normal(size=(N//2, 3))
        x_base /= np.linalg.norm(x_base, axis=1, keepdims=True)
        x_anti = np.vstack([x_base, -x_base])
        jitter = rng.normal(0, 0.05, (N, 3))
        x_anti = x_anti + jitter
        x_anti /= np.linalg.norm(x_anti, axis=1, keepdims=True)
        inits.append(x_anti)
        
    # Phase 1: L-BFGS-B optimization on each initialization
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
            
        try:
            x_opt, e_opt = _optimize_l_bfgs(x0, time_limit=12.0)
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt.copy()
        except Exception:
            continue
            
    if best_x is None:
        best_x = inits[0]
        best_e = _energy_eval(best_x)
        
    # Phase 2: BFGS tight polish on the best solution
    if time.time() < deadline - 40.0:
        try:
            x_final2, e_final2 = _optimize_bfgs_tight(best_x, time_limit=25.0)
            e_final2_precise = _energy_eval(x_final2)
            if e_final2_precise < best_e:
                best_x = x_final2
                best_e = e_final2_precise
        except Exception:
            pass
        
    # Phase 3: Ultra-tight L-BFGS-B polish (Newton-like)
    if time.time() < deadline - 15.0:
        try:
            x_final3, e_final3 = _polish_newton(best_x, time_limit=12.0)
            e_final3_precise = _energy_eval(x_final3)
            if e_final3_precise < best_e:
                best_x = x_final3
                best_e = e_final3_precise
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
