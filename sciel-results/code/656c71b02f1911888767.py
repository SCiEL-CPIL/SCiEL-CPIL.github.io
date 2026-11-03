import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for scipy
    x_flat = x.flatten()
    
    def energy_and_grad(x_flat, penalty_mu=0.0):
        """
        Computes the logarithmic energy and its gradient.
        If penalty_mu > 0, adds penalty for deviation from unit sphere.
        """
        x = x_flat.reshape(N, 3)
        
        # Compute pairwise distances
        # d[i, j] = ||x[i] - x[j]||
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(d_sq, 1.0)
        d = np.sqrt(d_sq)
        
        # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
        # We sum over all i != j and divide by 2
        # log(1/d) = -log(d)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        log_d = np.zeros_like(d)
        log_d[mask] = np.log(d[mask])
        energy = -0.5 * np.sum(log_d[mask])
        
        # Gradient of Energy
        # Grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
        # M_ij = 1/d_ij^2 for i!=j, 0 otherwise
        M = np.zeros_like(d_sq)
        M[mask] = 1.0 / d_sq[mask]
        
        # S_i = sum_j M_ij
        S = np.sum(M, axis=1)
        
        # Grad = M @ X - diag(S) @ X
        # M @ X is N x 3
        grad_eucl = M @ x - np.outer(S, x)
        
        if penalty_mu > 0:
            # Penalty: mu * sum (|x_i|^2 - 1)^2
            r_sq = np.sum(x * x, axis=1)
            penalty = penalty_mu * np.sum((r_sq - 1.0) ** 2)
            # Grad Penalty: 4 * mu * (r_sq - 1) * x
            grad_pen = 4.0 * penalty_mu * np.outer(r_sq - 1.0, x)
            
            total_energy = energy + penalty
            total_grad = grad_eucl + grad_pen
        else:
            total_energy = energy
            total_grad = grad_eucl
            
        return total_energy, total_grad.flatten()

    def objective(x_flat):
        e, _ = energy_and_grad(x_flat, penalty_mu=1000.0)
        return e

    def jac(x_flat):
        _, g = energy_and_grad(x_flat, penalty_mu=1000.0)
        return g

    # 2. L-BFGS-B Optimization with Penalty
    # We use a penalty to keep points on the sphere during the unconstrained optimization
    # This is a heuristic, but effective for this problem
    try:
        res = scipy.optimize.minimize(
            objective, 
            x_flat, 
            method='L-BFGS-B', 
            jac=jac, 
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x_opt = res.x.reshape(N, 3)
    except Exception:
        x_opt = x

    # Check time
    if time.time() > deadline - 5.0:
        # Project and return
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        return x_opt

    # 3. Riemannian Polish
    # Project to sphere
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    # Gradient Descent on Sphere
    # We use the Euclidean gradient projected onto the tangent space
    alpha = 0.01
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
            
        # Compute Euclidean Gradient
        _, grad_flat = energy_and_grad(x_opt.flatten(), penalty_mu=0.0)
        grad_eucl = grad_flat.reshape(N, 3)
        
        # Project to Tangent Space
        # g_tan = g - (g . x) x
        dot = np.sum(grad_eucl * x_opt, axis=1, keepdims=True)
        grad_tan = grad_eucl - dot * x_opt
        
        # Update
        x_new = x_opt + alpha * grad_tan
        
        # Normalize
        x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
        
        # Check Energy Decrease
        e_old, _ = energy_and_grad(x_opt.flatten(), penalty_mu=0.0)
        e_new, _ = energy_and_grad(x_new.flatten(), penalty_mu=0.0)
        
        if e_new < e_old:
            x_opt = x_new
            # Increase alpha slightly if improvement is good
            if e_old - e_new > 1e-8:
                alpha = min(alpha * 1.1, 0.1)
        else:
            # Decrease alpha
            alpha *= 0.5
            
    # Final projection
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    return x_opt
