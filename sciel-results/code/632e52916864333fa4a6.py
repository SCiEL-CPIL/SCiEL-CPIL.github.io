import numpy as np
from scipy.special import lpmv
import math

def _compute_y_values():
    """
    Precompute the values of the 21 real orthonormal spherical harmonics
    for degree l=10 at the quadrature points.
    Returns:
        Y: array of shape (N_phi, N_theta, 21)
        weights: array of shape (N_theta, N_phi) or flattened
    """
    l = 10
    N_theta = 70
    N_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta)
    x, w_gl = np.polynomial.legendre.leggauss(N_theta)
    
    # Phi points
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Create grids
    theta = np.arccos(x)
    # Meshgrid for broadcasting: (N_phi, N_theta)
    Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
    
    # Weights for the double integral
    weights = np.outer(np.ones(N_phi), w_gl) * (2 * np.pi / N_phi)
    
    # Compute harmonics
    Y = np.zeros((N_phi, N_theta, 21), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + math.lgamma(11 - abs_m) - math.lgamma(11 + abs_m))
        N_m = np.exp(log_N)
        
        P_lm = lpmv(abs_m, l, x)
        
        if m == 0:
            Y[:, :, m_idx] = N_m * P_lm
        elif m > 0:
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.cos(m * Phi)
        else: # m < 0
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * Phi)
            
    return Y, weights

# Precompute
Y_vals, weights = _compute_y_values()
Y_flat = Y_vals.flatten() # (N_points, 21)
w_flat = weights.flatten() # (N_points,)
N_points = Y_flat.shape[0]

def _compute_f_and_grad(c):
    """
    Computes f values, L4 norm, L2 norm, and the gradient of the objective J = L4/L2.
    """
    f = Y_flat @ c
    f4 = f ** 4
    f2 = f ** 2
    f3 = f ** 3
    
    integral_l4 = np.sum(w_flat * f4)
    integral_l2 = np.sum(w_flat * f2) # Should be dot(c,c) due to orthonormality, but let's use quadrature for consistency if needed? 
    # The task says L2 norm is sqrt(sum c^2). 
    l2_norm = np.sqrt(np.dot(c, c))
    
    # If L2 is 0, return inf
    if l2_norm < 1e-15:
        return None, None, None, None, 1e10, None

    l4_norm = integral_l4 ** 0.25
    
    # Objective J = l4_norm / l2_norm
    J = l4_norm / l2_norm
    
    # Gradient of J w.r.t c
    # J = A^{1/4} B^{-1/2}
    # dJ/dc_k = A^{-3/4} B^{-1/2} * (1/4 * 4 * sum(w * f^3 * Y_k)) - A^{1/4} B^{-3/2} * (1/2 * 2 * c_k)
    # dJ/dc_k = A^{-3/4} B^{-1/2} * sum(w * f^3 * Y_k) - A^{1/4} B^{-3/2} * c_k
    
    # Precompute terms
    # term1 = sum(w * f^3 * Y) -> vector of size 21
    term1 = Y_flat.T @ (w_flat * f3)
    
    # Coefficients
    coeff1 = (integral_l4 ** -0.75) * (l2_norm ** -1)
    coeff2 = (l4_norm) * (l2_norm ** -3)
    
    grad = coeff1 * term1 - coeff2 * c
    
    return f, l4_norm, l2_norm, J, grad, integral_l4

def _riemannian_cg(c0, max_iter=1000, tol=1e-10):
    """
    Riemannian Conjugate Gradient (Fletcher-Reeves) on the unit sphere.
    """
    c = c0 / np.linalg.norm(c0)
    prev_grad = None
    p = None
    beta = 0.0
    
    f, l4, l2, J, grad, _ = _compute_f_and_grad(c)
    if J is None:
        return c, 1e10
        
    # Project gradient to tangent space
    grad_tangent = grad - np.dot(grad, c) * c
    
    # Initial direction: negative gradient
    p = -grad_tangent
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-12:
        return c, J
    p = p / p_norm

    best_c = c
    best_J = J

    for i in range(max_iter):
        # Line search (Backtracking)
        alpha = 0.1
        current_J = J
        step = 1.0
        for _ in range(20):
            # Update
            c_new = c + alpha * p
            c_new_norm = np.linalg.norm(c_new)
            if c_new_norm < 1e-15:
                alpha *= 0.5
                continue
            c_new = c_new / c_new_norm
            
            _, _, _, J_new, _, _ = _compute_f_and_grad(c_new)
            
            if J_new < current_J:
                # Armijo condition
                if J_new < current_J - 0.5 * alpha * np.dot(grad_tangent, p):
                    break
            else:
                alpha *= 0.5
                step = 0.5
        else:
            # If no step found, break
            break
            
        c = c_new
        J = J_new
        if J < best_J:
            best_J = J
            best_c = c.copy()
            
        # Compute new gradient
        _, _, _, _, grad_new, _ = _compute_f_and_grad(c)
        if grad_new is None:
            break
            
        grad_new_tangent = grad_new - np.dot(grad_new, c) * c
        
        # Check convergence
        if np.linalg.norm(grad_new_tangent) < tol:
            break
            
        # Fletcher-Reeves beta
        if prev_grad is not None:
            beta = np.dot(grad_new_tangent, grad_new_tangent) / (np.dot(prev_grad, prev_grad) + 1e-15)
            # Ensure beta is non-negative for stability
            if beta < 0:
                beta = 0.0
        else:
            beta = 0.0
            
        p = -grad_new_tangent + beta * p
        
        # Re-project p to tangent space
        p = p - np.dot(p, c) * c
        
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            p = -grad_new_tangent
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                break
            p = p / p_norm
        else:
            p = p / p_norm
            
        prev_grad = grad_new_tangent
        grad_tangent = grad_new_tangent

    return best_c, best_J

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    n_coeffs = 21
    rng = np.random.default_rng(42)
    
    best_c = None
    best_val = np.inf
    
    n_restarts = 100 # Increase restarts as CG is fast
    
    for i in range(n_restarts):
        c0 = rng.normal(size=n_coeffs)
        c0 /= np.linalg.norm(c0)
        
        try:
            c_opt, val = _riemannian_cg(c0, max_iter=500)
            if val < best_val:
                best_val = val
                best_c = c_opt
        except Exception:
            pass
            
    if best_c is None:
        return np.ones(n_coeffs, dtype=np.float64)
        
    return best_c.astype(np.float64)
