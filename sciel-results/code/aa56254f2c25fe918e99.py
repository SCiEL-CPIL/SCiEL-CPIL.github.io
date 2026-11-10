import numpy as np
import time
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    
    Strategy: 
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. Reformulate as a constrained optimization on the sphere: minimize ∫ f^4 subject to ||c||_2 = 1.
    3. Use a projected gradient descent method with adaptive step size and momentum.
       The gradient is computed analytically.
       We project back to the sphere after each step.
    4. Use multiple diverse initializations and keep the best result.
    5. Use a deadline guard to ensure we return within 600s.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # --- 1. Precompute Spherical Harmonics ---
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Effective weights
    W_eff = np.repeat(w_gl, n_phi) * (2.0 * np.pi / n_phi)
    
    # Normalization constants
    N = np.zeros(n_coeffs)
    for m in range(0, k + 1):
        norm_factor = (2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        if m > 0:
            N[-m] = N[m]
            
    # Build Y matrix
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P_val = lpmv(0, k, x_flat)
            Y_matrix[:, m_idx] = N[0] * P_val
        elif m > 0:
            P_val = lpmv(m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P_val * np.cos(m * phi_flat)
        else:
            abs_m = -m
            P_val = lpmv(abs_m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * phi_flat)
            
    # --- 2. Define Objective and Gradient ---
    def objective_and_grad(c):
        """
        Compute objective J(c) = (∫ f^4) / (∫ f^2)^2 and its gradient.
        """
        f = Y_matrix @ c
        f2 = f * f
        D_val = np.dot(W_eff, f2)
        N_val = np.dot(W_eff, f2 * f2)
        
        if D_val < 1e-12:
            return 1e10, np.zeros_like(c)
            
        val = N_val / (D_val * D_val)
        
        # Gradient
        grad_N = 4.0 * (Y_matrix.T @ (W_eff * f**3))
        grad_D = 2.0 * (Y_matrix.T @ (W_eff * f))
        
        grad = (grad_N * D_val - 2.0 * N_val * grad_D) / (D_val ** 3)
        
        return val, grad

    # --- 3. Projected Gradient Descent with Momentum ---
    def projected_gradient_descent(x0, max_iter=5000, tol=1e-14):
        """
        Projected gradient descent on the sphere with adaptive step size.
        """
        c = x0.copy()
        c /= np.linalg.norm(c)
        
        momentum = np.zeros_like(c)
        beta = 0.9
        step_size = 0.1
        best_c = c.copy()
        best_val, _ = objective_and_grad(c)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            val, grad = objective_and_grad(c)
            
            if val < best_val:
                best_val = val
                best_c = c.copy()
            
            # Project gradient to tangent space
            grad_tangent = grad - np.dot(grad, c) * c
            
            # Adaptive step size with backtracking
            step = step_size
            for _ in range(20):
                c_new = c - step * grad_tangent
                c_new /= np.linalg.norm(c_new)
                val_new, _ = objective_and_grad(c_new)
                if val_new < val:
                    break
                step *= 0.5
            else:
                step = 0.1
                
            # Update with momentum
            momentum = beta * momentum + (1 - beta) * (-step * grad_tangent)
            c_new = c + momentum
            c_new /= np.linalg.norm(c_new)
            
            val_new, _ = objective_and_grad(c_new)
            if val_new < val:
                c = c_new
                step_size = min(step * 1.1, 0.5)
            else:
                step_size = max(step * 0.7, 0.01)
            
            if abs(val - best_val) < tol:
                break
        
        return best_c, best_val

    # --- 4. Multiple Initializations ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    inits = []
    
    # 1. Single modes
    for m in range(-k, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
    # 2. Symmetric pairs
    for m in range(1, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 1.0
        inits.append(init)
        
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = -1.0
        inits.append(init)
        
    # 3. Random vectors
    for _ in range(80):
        init = np.random.randn(n_coeffs)
        init /= np.linalg.norm(init)
        inits.append(init)
        
    # 4. Sparse random
    for _ in range(50):
        init = np.random.randn(n_coeffs)
        mask = np.random.rand(n_coeffs) < 0.3
        init[~mask] = 0
        if np.linalg.norm(init) > 1e-6:
            init /= np.linalg.norm(init)
        inits.append(init)
        
    np.random.shuffle(inits)
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            c_res, score = projected_gradient_descent(x0)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
                
        except Exception:
            continue

    # --- 5. Refine Best Solution with More Iterations ---
    if time.time() < deadline - 20:
        try:
            c_res, score = projected_gradient_descent(best_coeffs, max_iter=10000, tol=1e-16)
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
        except Exception:
            pass

    # --- 6. Validate and Return ---
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
