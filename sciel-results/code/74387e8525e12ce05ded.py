import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its tangent gradient.
    X: (N, 3) array of points.
    Returns:
        energy: float
        grad: (N, 3) array of tangent gradients.
    """
    # Normalize points to ensure they are on the sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    X = X / norms
    
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Distances
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero in log and grad
    # We need i < j for energy sum
    # Energy = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    
    # To compute efficiently:
    # Sum over all i != j is 2 * sum_{i<j}
    # So sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d)
    
    # Fill diagonal with 1.0 so log(1)=0
    np.fill_diagonal(dists, 1.0)
    
    # Clip distances to avoid log(0) if points are extremely close
    # The problem states points closer than 1e-9 make candidate invalid, 
    # but during optimization we might get close.
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)
    
    log_dists = np.log(safe_dists)
    np.fill_diagonal(log_dists, 0.0)
    
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient calculation
    # The Euclidean gradient of E = - sum_{i<j} log(|x_i - x_j|)
    # grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Note: This is the gradient of the function defined on R^3.
    # We need the gradient on the sphere, which is the projection of this gradient
    # onto the tangent space at x_i.
    # proj(v) = v - (v . x_i) x_i
    
    inv_dists_sq = 1.0 / (safe_dists ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad_euclid[i] = sum_j (x_i - x_j) * inv_dists_sq[i,j]
    grad_euclid = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    # dot_i = grad_euclid[i] . x_i
    dots = np.sum(grad_euclid * X, axis=1, keepdims=True)
    grad_tangent = grad_euclid - dots * X
    
    return energy, grad_tangent

def _wrapped_fun_and_grad(flat_x):
    """
    Wrapper for scipy.optimize.minimize.
    flat_x: (3*N,) array
    Returns:
        energy: float
        grad_flat: (3*N,) array
    """
    X = flat_x.reshape(N, 3)
    energy, grad = _energy_and_grad(X)
    return energy, grad.flatten()

def _optimize_lbfgs(X0, time_limit, max_iter=1000):
    """
    Optimize using L-BFGS-B.
    """
    X0 = X0.copy()
    norms = np.linalg.norm(X0, axis=1, keepdims=True)
    X0 = X0 / norms
    flat_x0 = X0.flatten()
    
    start_time = time.time()
    
    # We need a callback to check time
    def callback(xk):
        if time.time() - start_time > time_limit:
            # Raise exception to stop optimization
            raise TimeoutError
        return False

    try:
        res = minimize(
            _wrapped_fun_and_grad,
            flat_x0,
            method='L-BFGS-B',
            jac=True, # Since we return (f, g) from _wrapped_fun_and_grad? 
                      # No, minimize expects fun to return f, and jac to return g.
                      # Or fun can return (f, g) if jac=True? 
                      # Documentation: If jac is True, fun is assumed to return (f, g).
                      # Let's check scipy docs. 
                      # "If jac is True, fun is assumed to return (f, g)."
                      # My _wrapped_fun_and_grad returns (energy, grad_flat). This matches.
            callback=callback,
            maxiter=max_iter,
            options={'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
    except TimeoutError:
        # If timeout, use the best x found so far?
        # L-BFGS-B might not keep the best x easily accessible on exception.
        # It's better to just catch and return the current X0 or the last known good one.
        # However, minimize usually updates xk in callback. 
        # To be safe, we can just return the initial if timeout happens too early, 
        # or we can modify the strategy to check time outside.
        # Given the speed of L-BFGS-B, timeout is unlikely for short runs.
        # Let's assume we don't hit the timeout inside the loop for short runs.
        # If we do, we return the initial points.
        return X0, _energy_and_grad(X0)[0]
    except Exception as e:
        # If any other error, return initial
        return X0, _energy_and_grad(X0)[0]
    
    if res.success or hasattr(res, 'x'):
        X_opt = res.x.reshape(N, 3)
        # Final normalization
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        X_opt = X_opt / norms
        energy = _energy_and_grad(X_opt)[0]
        return X_opt, energy
    else:
        return X0, _energy_and_grad(X0)[0]

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Deadline: 150s wall clock. We aim for 140s.
    deadline = time.time() + 140.0
    
    best_X = None
    best_energy = float('inf')
    
    # Strategy: Multiple restarts with L-BFGS-B
    # 1. Fibonacci sphere
    # 2. Random points (multiple seeds)
    # 3. Fibonacci + noise
    
    initializers = []
    
    # Fibonacci
    X_fib = _fibonacci_sphere(N)
    initializers.append(X_fib)
    
    # Random
    rng = np.random.default_rng(42)
    for seed in range(10):
        r = np.random.default_rng(seed)
        X_rand = r.random((N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        initializers.append(X_rand)
        
    # Fibonacci + noise
    for noise_level in [0.05, 0.1, 0.2, 0.5]:
        X_noise = X_fib + rng.normal(0, noise_level, (N, 3))
        X_noise = X_noise / np.linalg.norm(X_noise, axis=1, keepdims=True)
        initializers.append(X_noise)
        
    # Run optimization for each initializer
    for i, X0 in enumerate(initializers):
        if time.time() > deadline - 10.0: # Leave 10s buffer
            break
            
        remaining = deadline - time.time()
        # Allocate time for this run. 
        # L-BFGS-B is fast, so 5-10 seconds per run is plenty.
        # If we have 15 initializers, 140s / 15 = 9s each.
        time_budget = min(10.0, remaining * 0.9)
        
        if time_budget < 1.0:
            break
            
        X_opt, energy = _optimize_lbfgs(X0, time_budget, max_iter=200)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt.copy()
            
        # Print progress for debugging (optional, but good for local runs)
        # print(f"Init {i}: Energy {energy:.6f}")

    # If we still have time, polish the best result with more iterations
    if best_X is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        # Use a larger max_iter for the final polish
        X_opt, energy = _optimize_lbfgs(best_X, remaining, max_iter=500)
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt
            
    # Final normalization and check
    if best_X is None:
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N)
    else:
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / norms
        
    # Ensure no points are too close
    # The problem says points closer than 1e-9 make candidate invalid.
    # L-BFGS-B usually avoids this, but let's check.
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    
    if min_dist < 1e-9:
        # If points are too close, we need to perturb them.
        # This is a safety net.
        rng = np.random.default_rng(0)
        for _ in range(100):
            # Perturb the closest pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i == j: continue
            # Push them apart
            v = best_X[i] - best_X[j]
            v = v / np.linalg.norm(v)
            best_X[i] += 1e-6 * v
            best_X[j] -= 1e-6 * v
            # Re-normalize
            best_X[i] /= np.linalg.norm(best_X[i])
            best_X[j] /= np.linalg.norm(best_X[j])
            
            diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            min_dist = np.min(dists)
            if min_dist > 1e-9:
                break
                
    return best_X
