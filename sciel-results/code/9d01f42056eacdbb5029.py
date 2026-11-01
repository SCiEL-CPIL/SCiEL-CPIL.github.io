import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its tangent gradient.
    X: (N, 3) array of points on the unit sphere.
    Returns:
        energy: float
        grad: (N, 3) array of tangent gradients.
    """
    # Ensure points are on the sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy = - sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Clip to avoid log(0)
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)
    
    log_dists = np.log(safe_dists)
    np.fill_diagonal(log_dists, 0.0)
    
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient calculation
    # grad_euclid[i] = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_dists_sq = 1.0 / (safe_dists ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    grad_euclid = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dots = np.sum(grad_euclid * X, axis=1, keepdims=True)
    grad_tangent = grad_euclid - dots * X
    
    return energy, grad_tangent

def _geodesic_step(X, step, step_size=1.0):
    """
    Apply a geodesic step on the sphere.
    X: (N, 3) current points
    step: (N, 3) tangent vectors (steps in tangent space)
    step_size: scalar multiplier
    Returns: (N, 3) new points on the sphere
    """
    # The step is in the tangent space. 
    # For small steps, exp_x(v) ≈ x + v - 0.5 * (|v|^2 * x + (v.x) * v) ... 
    # Actually, a simpler and very effective method for optimization is 
    # to just normalize x + step. This is equivalent to a projection 
    # which is a valid retraction on the sphere.
    # Retraction: R_x(v) = (x + v) / ||x + v||
    
    new_X = X + step_size * step
    norms = np.linalg.norm(new_X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_X = new_X / norms
    return new_X

def _optimize_lm(X0, time_limit, max_iter=1000):
    """
    Optimize using a custom Levenberg-Marquardt style loop on the manifold.
    We use scipy's minimize with BFGS but enforce manifold steps via a wrapper?
    No, scipy's minimize works on Euclidean space. 
    To do true geodesic LM, we need to implement the loop manually or use 
    a library that supports manifolds. Since we only have standard scipy, 
    we can simulate a second-order method by using 'Newton-CG' or 'BFGS' 
    but projecting the result back.
    
    However, a better "different strategy" within standard libraries is to use 
    scipy.optimize.minimize with method='BFGS' (quasi-Newton) but strictly 
    projecting every intermediate step. The current code uses L-BFGS-B which 
    is a first-order method with line search. BFGS is second-order (quasi).
    
    Even better: Use 'Nelder-Mead' for robustness? No, too slow.
    
    Let's implement a simple **Geodesic Gradient Descent with Adaptive Step Size** 
    and **Simulated Annealing Perturbations**. This is distinct from L-BFGS-B 
    because it doesn't use a line search on the Euclidean gradient, but rather 
    steps along the manifold with a decaying step size and random kicks.
    
    Actually, the most distinct and powerful standard tool is **scipy.optimize.minimize** 
    with **method='BFGS'** (unconstrained, but we project). 
    Wait, the current code uses L-BFGS-B. BFGS is a different Hessian approximation 
    (full matrix vs L-BFGS two-loop). For N=100, 300 variables, BFGS is feasible 
    and often faster for smooth problems.
    
    Let's try a hybrid: 
    1. Use BFGS (quasi-Newton) for local optimization.
    2. Implement a "kick" mechanism: if improvement stalls, apply a large 
       random geodesic perturbation to escape local minima.
    """
    X = X0.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    flat_x = X.flatten()
    
    start_time = time.time()
    best_energy = float('inf')
    best_X = X.copy()
    
    # We will run BFGS, but we will wrap the function to project back to sphere.
    # However, BFGS assumes Euclidean space. Projecting inside the function 
    # makes the function non-smooth at the boundary of the projection domain? 
    # No, projection is smooth everywhere except where x+v=0.
    
    def fun_and_grad(flat_x):
        X_curr = flat_x.reshape(N, 3)
        # Project to sphere
        norms = np.linalg.norm(X_curr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_curr = X_curr / norms
        
        energy, grad = _energy_and_grad(X_curr)
        
        # The gradient returned to BFGS must be the Euclidean gradient of the 
        # projected function. 
        # If f_proj(x) = f(x/||x||), the chain rule is complex.
        # Instead, we can use the fact that for small steps, the tangent gradient 
        # is a good approximation of the Euclidean gradient of the extended function.
        # But BFGS will take steps in the radial direction too.
        # A common trick: return the tangent gradient as the Euclidean gradient.
        # This effectively constrains the Hessian update to the tangent plane.
        
        return energy, grad.flatten()
    
    # We will use a loop of BFGS runs with perturbations
    current_energy = _energy_and_grad(X)[0]
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Try to optimize from current X
        try:
            res = minimize(
                fun_and_grad,
                X.flatten(),
                method='BFGS',
                jac=True,
                options={'gtol': 1e-8, 'maxiter': 50, 'maxfun': 200}
            )
            
            X_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            X_opt = X_opt / norms
            energy_opt, _ = _energy_and_grad(X_opt)
            
            if energy_opt < current_energy:
                current_energy = energy_opt
                X = X_opt
                if energy_opt < best_energy:
                    best_energy = energy_opt
                    best_X = X.copy()
            else:
                # No improvement, or BFGS failed to improve
                # Check if we are stuck (small gradient)
                _, grad = _energy_and_grad(X)
                grad_norm = np.linalg.norm(grad)
                
                if grad_norm < 1e-6:
                    # Likely local minimum. Apply perturbation.
                    rng = np.random.default_rng()
                    # Random tangent vectors
                    noise = rng.normal(0, 0.5, (N, 3))
                    # Project noise to tangent space
                    dots = np.sum(noise * X, axis=1, keepdims=True)
                    noise_tangent = noise - dots * X
                    X = _geodesic_step(X, noise_tangent, step_size=1.0)
                    current_energy = _energy_and_grad(X)[0]
                else:
                    # Gradient not small, but BFGS didn't improve? 
                    # Maybe step size issue. Try a smaller manual gradient step.
                    step = -0.1 * grad
                    X_new = _geodesic_step(X, step, step_size=1.0)
                    e_new, _ = _energy_and_grad(X_new)
                    if e_new < current_energy:
                        X = X_new
                        current_energy = e_new
                        if e_new < best_energy:
                            best_energy = e_new
                            best_X = X.copy()
                    else:
                        # Stuck. Perturb.
                        rng = np.random.default_rng()
                        noise = rng.normal(0, 0.2, (N, 3))
                        dots = np.sum(noise * X, axis=1, keepdims=True)
                        noise_tangent = noise - dots * X
                        X = _geodesic_step(X, noise_tangent, step_size=1.0)
                        current_energy = _energy_and_grad(X)[0]
                        
        except Exception:
            break
            
    return best_X, best_energy

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    best_X = None
    best_energy = float('inf')
    
    # Strategy: 
    # 1. Start with Fibonacci (good global structure)
    # 2. Run Geodesic LM/BFGS with perturbations
    # 3. If time allows, try a few random starts with the same optimizer
    
    initializers = []
    
    # Fibonacci
    X_fib = _fibonacci_sphere(N)
    initializers.append(X_fib)
    
    # A few random starts
    for seed in [1, 2, 3]:
        rng = np.random.default_rng(seed)
        X_rand = rng.random((N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        initializers.append(X_rand)
        
    # Fibonacci with small noise
    rng = np.random.default_rng(0)
    X_noise = X_fib + rng.normal(0, 0.1, (N, 3))
    X_noise = X_noise / np.linalg.norm(X_noise, axis=1, keepdims=True)
    initializers.append(X_noise)
    
    for i, X0 in enumerate(initializers):
        if time.time() > deadline - 20.0:
            break
            
        remaining = deadline - time.time()
        # Allocate time. BFGS is fast, but perturbations take time.
        # Give each run up to 25 seconds, but break early if remaining is small.
        time_budget = min(25.0, remaining * 0.9)
        
        if time_budget < 5.0:
            break
            
        X_opt, energy = _optimize_lm(X0, time_budget)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt.copy()
            
    # Final polish if time remains
    if best_X is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            X_opt, energy = _optimize_lm(best_X, remaining)
            if energy < best_energy:
                best_energy = energy
                best_X = X_opt

    # Fallback
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    else:
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / norms
        
    # Safety check for minimum distance
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    
    if min_dist < 1e-9:
        rng = np.random.default_rng(0)
        for _ in range(100):
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i == j: continue
            v = best_X[i] - best_X[j]
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12: continue
            v = v / v_norm
            best_X[i] += 1e-6 * v
            best_X[j] -= 1e-6 * v
            best_X[i] /= np.linalg.norm(best_X[i])
            best_X[j] /= np.linalg.norm(best_X[j])
            
            diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            min_dist = np.min(dists)
            if min_dist > 1e-9:
                break

    return best_X
