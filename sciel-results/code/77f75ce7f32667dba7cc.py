import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio_and_grad(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to coeffs.
    f = Y @ coeffs
    ||f||_2 = sqrt(sum w f^2)
    ||f||_4 = (sum w f^4)^(1/4)
    
    Let A = sum w f^2 = ||f||_2^2
    Let B = sum w f^4 = ||f||_4^4
    
    R = B^(1/4) / A^(1/2)
    
    dR/da = (1/4) B^(-3/4) (sum w 4 f^3 df) / (A^(1/2)) - B^(1/4) (1/2) A^(-3/2) (sum w 2 f df)
    
    Since f = Y c, df = Y dc.
    grad_R = (1/4) B^(-3/4) A^(-1/2) Y^T (4 w f^3) - B^(1/4) (1/2) A^(-3/2) Y^T (2 w f)
    grad_R = B^(-3/4) A^(-1/2) Y^T (w f^3) - B^(1/4) A^(-3/2) Y^T (w f)
    """
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    
    A = np.dot(weights, f2) # ||f||_2^2
    B = np.dot(weights, f4) # ||f||_4^4
    
    if A < 1e-15 or B < 1e-15:
        return 1e10, np.zeros_like(coeffs)
    
    R = np.sqrt(np.sqrt(B)) / np.sqrt(A)
    
    # Gradient terms
    term1 = Y.T @ (weights * f**3)
    term2 = Y.T @ (weights * f)
    
    grad = (B**(-0.75)) * (A**(-0.5)) * term1 - (B**0.25) * (A**(-1.5)) * term2
    
    return R, grad

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    # Strategy: Gradient-based optimization with multiple diverse starts.
    # We search the full 21-dimensional space but use symmetry to generate good starts.
    # The objective is smooth and differentiable.
    
    best_coeffs = None
    best_val = np.inf
    
    # Generate diverse initial points
    # 1. Pure m=0
    # 2. Pure m=10 (cos)
    # 3. Random points on the unit sphere in coefficient space
    # 4. Points exploiting z-symmetry (only even m)
    
    rng = np.random.default_rng(42)
    
    initial_points = []
    
    # Start 1: m=0
    c0 = np.zeros(21)
    c0[10] = 1.0
    initial_points.append(c0)
    
    # Start 2: m=10 cos
    c1 = np.zeros(21)
    c1[20] = 1.0
    initial_points.append(c1)
    
    # Start 3: m=10 sin
    c2 = np.zeros(21)
    c2[10-10] = 1.0 # m=-10
    initial_points.append(c2)
    
    # Start 4-20: Random points
    for _ in range(17):
        c_rand = rng.normal(size=21)
        c_rand /= np.linalg.norm(c_rand)
        initial_points.append(c_rand)
        
    # Start 21-30: Z-symmetric random points (only even m)
    even_m_indices = [10, 12, 14, 16, 18, 20] # m = 0, 2, 4, 6, 8, 10
    for _ in range(10):
        c_sym = np.zeros(21)
        vals = rng.normal(size=6)
        for i, idx in enumerate(even_m_indices):
            c_sym[idx] = vals[i]
        norm = np.linalg.norm(c_sym)
        if norm > 1e-9:
            c_sym /= norm
        initial_points.append(c_sym)
        
    # Optimize each start
    for i, c_init in enumerate(initial_points):
        if time.time() > deadline - 10.0:
            break
            
        def objective_and_grad(c):
            val, grad = _compute_ratio_and_grad(c, Y, weights)
            return val, grad
        
        try:
            # Use BFGS with exact gradient
            res = minimize(
                objective_and_grad, 
                c_init, 
                method='L-BFGS-B', 
                jac=True, # Since we return tuple (f, grad)
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
                
        except Exception as e:
            pass
            
    # If we have time, do a second pass with more iterations on the best found
    if best_coeffs is not None and time.time() < deadline - 20.0:
        def objective_and_grad(c):
            val, grad = _compute_ratio_and_grad(c, Y, weights)
            return val, grad
            
        try:
            res = minimize(
                objective_and_grad, 
                best_coeffs, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
            
    # Final validation and scaling
    if best_coeffs is None:
        # Fallback
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0
        
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    
    # Ensure valid range
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs * 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1.0
        
    return best_coeffs
