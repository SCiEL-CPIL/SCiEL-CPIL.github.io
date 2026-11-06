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
    # For E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # On the sphere, project to tangent: grad_tan = grad_euclid - (grad_euclid·x_i) x_i
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    # Project to tangent space
    g_dot_x = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tangent = grad_euclid - g_dot_x * x
    
    return energy, grad_tangent.flatten()

def _riemannian_grad_descent(x0, max_iter=5000, time_limit=20.0, global_deadline=None):
    """
    Riemannian gradient descent with adaptive step size and backtracking.
    Uses unnormalized gradient with proper scaling.
    """
    start_time = time.time()
    x = x0.copy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    # Initial step size based on gradient magnitude
    energy, grad = _objective_and_grad(x.flatten())
    g_norm = np.linalg.norm(grad)
    if g_norm < 1e-14:
        return x, energy
    
    lr = 0.5 / g_norm  # scale so that initial step is moderate
    best_energy = energy
    best_x = x.copy()
    patience = 0
    max_patience = 300
    
    for it in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        if global_deadline is not None and time.time() > global_deadline - 35.0:
            break
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Backtracking line search with Armijo condition
        alpha = lr
        success = False
        c1 = 1e-4  # Armijo parameter
        for _ in range(15):
            x_new = x - alpha * grad.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            
            e_new, _ = _objective_and_grad(x_new.flatten())
            if e_new < energy - c1 * alpha * g_norm * g_norm:
                x = x_new
                energy = e_new
                success = True
                break
            alpha *= 0.5
        
        if not success:
            # Try a very small step
            alpha = 1e-6 / max(g_norm, 1e-10)
            x_new = x - alpha * grad.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            e_new, _ = _objective_and_grad(x_new.flatten())
            if e_new < energy:
                x = x_new
                energy = e_new
            else:
                break
        
        # Recompute gradient for next iteration
        energy, grad = _objective_and_grad(x.flatten())
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-12:
            break
        
        # Adapt learning rate
        if success and alpha > lr * 0.3:
            lr = min(lr * 1.1, 2.0 / max(g_norm, 1e-10))
        else:
            lr = max(lr * 0.8, 0.1 / max(g_norm, 1e-10))
    
    return best_x, best_energy

def _optimize_l_bfgs(x0, time_limit=15.0, global_deadline=None):
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
            'maxiter': 3000,
            'ftol': 1e-15,
            'gtol': 1e-10,
            'maxls': 40,
            'maxcor': 40,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _optimize_bfgs_tight(x0, time_limit=25.0, global_deadline=None):
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
            'maxiter': 2000,
            'gtol': 1e-12,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _polish_newton(x0, time_limit=20.0, global_deadline=None):
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
            'maxiter': 2000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 50,
            'maxcor': 40,
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
    
    # 2. Perturbed Fibonacci Spirals
    for scale in [0.05, 0.1, 0.2, 0.5]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # 3. Random Initializations
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
    
    # 5. Clustered initialization
    for _ in range(2):
        centers = rng.normal(size=(15, 3))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        x_clustered = np.array([centers[i % 15] + rng.normal(0, 0.1, 3) for i in range(N)])
        x_clustered /= np.linalg.norm(x_clustered, axis=1, keepdims=True)
        inits.append(x_clustered)
        
    # Phase 1: Riemannian gradient descent on each initialization
    # Use shorter time per init to cover more starts
    rgd_time = 6.0
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 55.0:
            break
            
        try:
            x_opt, e_opt = _riemannian_grad_descent(x0, max_iter=3000, time_limit=rgd_time, global_deadline=deadline)
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt.copy()
        except Exception:
            continue
            
    if best_x is None:
        best_x = inits[0]
        best_e = _energy_eval(best_x)
        
    # Phase 2: L-BFGS-B optimization on the best solution
    if time.time() < deadline - 45.0:
        try:
            x_final2, e_final2 = _optimize_l_bfgs(best_x, time_limit=20.0, global_deadline=deadline)
            e_final2_precise = _energy_eval(x_final2)
            if e_final2_precise < best_e:
                best_x = x_final2
                best_e = e_final2_precise
        except Exception:
            pass
        
    # Phase 3: BFGS tight polish
    if time.time() < deadline - 30.0:
        try:
            x_final3, e_final3 = _optimize_bfgs_tight(best_x, time_limit=15.0, global_deadline=deadline)
            e_final3_precise = _energy_eval(x_final3)
            if e_final3_precise < best_e:
                best_x = x_final3
                best_e = e_final3_precise
        except Exception:
            pass
        
    # Phase 4: Ultra-tight L-BFGS-B polish (Newton-like)
    if time.time() < deadline - 15.0:
        try:
            x_final4, e_final4 = _polish_newton(best_x, time_limit=10.0, global_deadline=deadline)
            e_final4_precise = _energy_eval(x_final4)
            if e_final4_precise < best_e:
                best_x = x_final4
                best_e = e_final4_precise
        except Exception:
            pass
    
    # Phase 5: Final micro-polish with even tighter settings if time permits
    if time.time() < deadline - 5.0:
        try:
            x0_flat = best_x.flatten()
            res = minimize(
                _objective_and_grad, 
                x0_flat, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-17,
                    'gtol': 1e-15,
                    'maxls': 50,
                    'maxcor': 40,
                    'disp': False
                }
            )
            x_final5 = res.x.reshape(N, 3)
            norms = np.linalg.norm(x_final5, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            x_final5 = x_final5 / norms
            e_final5_precise = _energy_eval(x_final5)
            if e_final5_precise < best_e:
                best_x = x_final5
                best_e = e_final5_precise
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
