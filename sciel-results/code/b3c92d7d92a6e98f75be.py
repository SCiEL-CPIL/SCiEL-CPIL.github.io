# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Minimizes the L4/L2 norm ratio for degree-10 spherical harmonics using 
    multi-start L-BFGS-B optimization with analytic gradients.
    Returns the optimal coefficients c_m for m = -10..10.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    cos_theta_vals = x_gl
    sin_theta_vals = np.sqrt(np.maximum(0, 1 - cos_theta_vals**2))
    
    # 2. Pre-compute Basis Functions Y_m
    # Y_m values at all grid points. Shape: (n_coeffs, n_grid)
    n_grid = 70 * n_phi
    Y = np.zeros((n_coeffs, n_grid), dtype=np.float64)
    
    # Quadrature weights for the surface integral
    # Integral over sphere: sum_{i,j} f(x_i, phi_j) w_gl_i dphi
    # Weights shape: (n_grid,)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    for m_idx in range(n_coeffs):
        m = m_idx - k
        abs_m = abs(m)
        
        # Normalization constant N_m
        # N_m = sqrt( (21)/(4*pi) * (10-m)! / (10+m)! )
        # Note: The formula in the prompt uses (10-m)!/(10+m)! which depends on |m|
        num = factorial(10 - abs_m)
        den = factorial(10 + abs_m)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        # Associated Legendre Polynomial P_10^|m|(cos_theta)
        P_vals = lpmv(abs_m, 10, cos_theta_vals)
        
        if m == 0:
            Y_vals = np.tile(P_vals[:, np.newaxis], (1, n_phi)) * N_m
        elif m > 0:
            cos_term = np.cos(m * phi)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * cos_term[np.newaxis, :])
        else:
            sin_term = np.sin(abs_m * phi)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * sin_term[np.newaxis, :])
        
        Y[m_idx, :] = Y_vals.flatten()

    # 3. Define Objective and Gradient
    # We want to minimize R(c) = ||f||_4 / ||f||_2
    # R(c) = ( \int f^4 )^{1/4} / ( \int f^2 )^{1/2}
    # Let A = \int f^2, B = \int f^4.
    # R = B^{1/4} A^{-1/2}
    # log R = 1/4 log B - 1/2 log A
    # grad log R = 1/4 (grad B / B) - 1/2 (grad A / A)
    # grad R = R * grad log R
    
    def objective_and_grad(c):
        # c is shape (n_coeffs,)
        # f = Y.T @ c
        f = Y.T @ c
        
        f_sq = f**2
        f_four = f_sq**2
        
        A = np.sum(weights * f_sq)  # ||f||_2^2
        B = np.sum(weights * f_four) # ||f||_4^4
        
        if A < 1e-15:
            return 1e10, np.zeros_like(c)
            
        R = (B ** 0.25) / (A ** 0.5)
        
        # Gradient of A = \int f^2 dOmega
        # dA/dc_j = \int 2 f Y_j dOmega = 2 \sum_k w_k f_k Y_j(k)
        # grad A = 2 Y @ (weights * f)
        grad_A = 2.0 * (Y @ (weights * f))
        
        # Gradient of B = \int f^4 dOmega
        # dB/dc_j = \int 4 f^3 Y_j dOmega = 4 \sum_k w_k f_k^3 Y_j(k)
        # grad B = 4 Y @ (weights * f^3)
        grad_B = 4.0 * (Y @ (weights * f**3))
        
        # grad R = R * (1/4 * grad_B / B - 1/2 * grad_A / A)
        grad_R = R * (0.25 * grad_B / B - 0.5 * grad_A / A)
        
        return R, grad_R

    # 4. Multi-start Optimization
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 5.0
    
    best_R = np.inf
    best_c = np.ones(n_coeffs, dtype=np.float64) / np.sqrt(n_coeffs)
    
    # Initial candidates
    # 1. All ones (normalized)
    c0 = np.ones(n_coeffs, dtype=np.float64)
    c0 /= np.linalg.norm(c0)
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    starts = [c0]
    for _ in range(15):
        c_rand = rng.normal(0, 1, n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)
        
    # 3. Some structured starts (e.g., only m=0, only m=10, etc.)
    for m_idx in [0, 5, 10, 15, 20]:
        c_struct = np.zeros(n_coeffs)
        c_struct[m_idx] = 1.0
        starts.append(c_struct)

    for c_init in starts:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            res = minimize(
                lambda c: objective_and_grad(c)[0], 
                c_init, 
                jac=lambda c: objective_and_grad(c)[1], 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            R_res = res.fun
            if R_res < best_R:
                best_R = R_res
                best_c = res.x
        except Exception:
            continue

    # 5. Refine the best result found with a tighter tolerance if time permits
    if time.time() - start_time < time_limit - safety_margin - 10:
        try:
            res = minimize(
                lambda c: objective_and_grad(c)[0], 
                best_c, 
                jac=lambda c: objective_and_grad(c)[1], 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'maxls': 100, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            R_res = res.fun
            if R_res < best_R:
                best_R = R_res
                best_c = res.x
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_coeffs, dtype=np.float64)
        
    # Ensure max|c| is within bounds (1e-3, 1e5)
    # Since we don't strictly constrain norm in the optimizer, it might drift.
    # But R is scale invariant. The optimizer might find a direction but scale arbitrarily.
    # L-BFGS-B on a scale-invariant function can sometimes drift in scale.
    # Let's normalize the final result to a reasonable scale, e.g., unit L2 norm of coefficients?
    # Or just check bounds.
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
        
    return best_c

# EVOLVE-BLOCK-END
