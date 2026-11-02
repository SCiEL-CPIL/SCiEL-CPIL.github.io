import numpy as np
import time
import math

N = 100

def _energy_and_grad(X):
    """
    Calculate the logarithmic energy and the Euclidean gradient.
    X: (N, 3) array of points on the sphere.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise distances
    # d[i, j] = ||x_i - x_j||
    # We need sum over i < j
    # d^2 = 2 - 2 * (x_i . x_j)
    # d = sqrt(2 - 2 * dot)
    
    # Efficient computation of distances
    # dots = X @ X.T
    # d2 = 2 - 2 * dots
    # np.fill_diagonal(d2, 1.0) # Avoid division by zero, though we mask it
    # d = np.sqrt(d2)
    
    # Gradient formula:
    # grad_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: d_ij^2 = 2 - 2 * dots[i, j]
    
    dots = X @ X.T
    # Ensure numerical stability for d2
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0) # Placeholder to avoid 0 division, will be masked
    
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # We can compute sum over all i!=j and divide by 2
    # log(d) = 0.5 * log(d2)
    # sum_{i!=j} log(d_ij) = 0.5 * sum_{i!=j} log(d2_ij)
    # We need to exclude diagonal (d2=1, log=0)
    # mask = 1 - np.eye(N)
    # energy = -0.5 * np.sum(mask * np.log(d2))
    # But careful: log(1/d) = -log(d).
    # E = sum_{i<j} -log(d_ij)
    # Let S = sum_{i!=j} -log(d_ij)
    # E = S / 2
    
    # For gradient:
    # grad_i = - sum_{j != i} (x_i - x_j) / d2_ij
    # Let A be a matrix where A[i, j] = 1 / d2_ij if i!=j, else 0.
    # Then grad_i = - sum_j A[i, j] (x_i - x_j)
    # grad_i = - (sum_j A[i, j] x_i) + (sum_j A[i, j] x_j)
    # grad_i = - x_i * (sum_j A[i, j]) + (A @ X)[i]
    
    # Create A
    A = 1.0 / d2
    np.fill_diagonal(A, 0.0)
    
    # Energy calculation
    # Use log(d2) for stability? log(d) = 0.5 log(d2)
    # E = -0.5 * sum_{i!=j} 0.5 log(d2_ij) ? No.
    # log(1/d) = -log(d) = -0.5 log(d2)
    # E = sum_{i<j} -0.5 log(d2_ij)
    # Sum_{i<j} is half of Sum_{i!=j}
    # E = 0.5 * Sum_{i!=j} -0.5 log(d2_ij) = -0.25 * Sum_{i!=j} log(d2_ij)
    
    # Let's stick to explicit sum for clarity in gradient derivation, but vectorize.
    # Energy
    log_d2 = np.log(d2)
    np.fill_diagonal(log_d2, 0.0)
    energy = -0.25 * np.sum(log_d2)
    
    # Gradient
    # row_sums = np.sum(A, axis=1)
    # term1 = X * row_sums[:, np.newaxis]
    # term2 = A @ X
    # grad = term2 - term1
    # Wait, formula: grad_i = - sum_j A[i, j] (x_i - x_j)
    # = - sum_j A[i, j] x_i + sum_j A[i, j] x_j
    # = - x_i (sum_j A[i, j]) + (A @ X)_i
    row_sums = np.sum(A, axis=1)
    grad = (A @ X) - X * row_sums[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return X / norms

def _optimize(X0, max_iters=5000, tol=1e-10, time_limit=None, start_time=None):
    """
    Riemannian Gradient Descent with backtracking line search.
    """
    if start_time is None:
        start_time = time.time()
    
    X = _project_to_sphere(X0.copy())
    energy, grad = _energy_and_grad(X)
    
    # Initial step size
    alpha = 0.01
    
    for it in range(max_iters):
        if time_limit and (time.time() - start_time > time_limit):
            break
            
        # Project gradient to tangent space
        # P(g) = g - (g . x) x
        dot_gx = np.sum(grad * X, axis=1, keepdims=True)
        grad_tangent = grad - dot_gx * X
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
            
        # Backtracking line search
        # We want to minimize f(X)
        # f_new = f(X - alpha * grad_tangent) after projection
        # Armijo condition: f_new <= f + c * alpha * grad^T (X_new - X)
        # Approximation: f_new <= f + c * alpha * ||grad_tangent||^2 (negative direction)
        # Actually, directional derivative is -||grad_tangent||^2
        # So f_new <= f - c * alpha * ||grad_tangent||^2
        
        c = 1e-4
        rho = 0.5
        improved = False
        for _ in range(50):
            X_trial = X - alpha * grad_tangent
            X_trial = _project_to_sphere(X_trial)
            
            energy_trial, _ = _energy_and_grad(X_trial)
            
            # Check if energy decreased sufficiently
            # Theoretical decrease is alpha * ||grad_tangent||^2
            # We require energy_trial <= energy - c * alpha * (grad_tangent^T grad_tangent)
            # grad_tangent^T grad_tangent = sum(||g_t_i||^2)
            grad_sq_norm = np.sum(grad_tangent**2)
            
            if energy_trial <= energy - c * alpha * grad_sq_norm:
                X = X_trial
                energy = energy_trial
                improved = True
                break
            else:
                alpha *= rho
        
        if not improved:
            # If we couldn't improve, reduce alpha further or break
            alpha *= 0.1
            if alpha < 1e-12:
                break
            
        # Update gradient for next iteration
        _, grad = _energy_and_grad(X)

    return X, energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    best_energy = float('inf')
    best_X = None
    
    # Generate diverse initial points
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    initializations = [X_fib]
    
    # 2. Random points (multiple seeds)
    rng = np.random.default_rng(42)
    for _ in range(5):
        X_rand = rng.normal(size=(N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        initializations.append(X_rand)
        
    # 3. Perturbed Fibonacci
    for _ in range(5):
        X_pert = X_fib + 0.1 * rng.normal(size=(N, 3))
        X_pert = X_pert / np.linalg.norm(X_pert, axis=1, keepdims=True)
        initializations.append(X_pert)
        
    # Optimize each initialization
    # We will run them sequentially, checking time.
    # If we have time, we can run more iterations for the best one found so far.
    
    current_best_energy = float('inf')
    current_best_X = None
    
    for k, X0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            # Not enough time for a full run, skip or do a quick polish
            # If this is the last one, maybe do a quick run?
            # For now, break to ensure we return something.
            break
            
        # Determine time limit for this specific run
        # Allow 10s per run roughly, but ensure we finish before deadline
        time_left = deadline - time.time()
        run_limit = min(10.0, time_left - 5.0) # Leave 5s buffer
        
        X_opt, E_opt = _optimize(X0, max_iters=2000, tol=1e-8, time_limit=run_limit, start_time=time.time())
        
        if E_opt < current_best_energy:
            current_best_energy = E_opt
            current_best_X = X_opt
            
    # If we have time left, try to polish the best solution found
    if current_best_X is not None and time.time() < deadline - 5.0:
        # Run a more precise optimization on the best candidate
        # Use smaller tolerance and more iterations
        time_left = deadline - time.time() - 2.0
        X_polished, E_polished = _optimize(current_best_X, max_iters=5000, tol=1e-10, time_limit=time_left, start_time=time.time())
        
        if E_polished < current_best_energy:
            current_best_energy = E_polished
            current_best_X = X_polished
            
    # If we didn't find any (shouldn't happen), return Fibonacci
    if current_best_X is None:
        current_best_X = X_fib
        
    # Final check: ensure points are on sphere and valid
    current_best_X = _project_to_sphere(current_best_X)
    
    return current_best_X
