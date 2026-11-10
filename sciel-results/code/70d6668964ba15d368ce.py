# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_{m=-10}^{10} c_m Y_m that minimizes the L4/L2 norm ratio.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with real coefficients c_m ordered m = -10..10
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s deadline with safety margin
    
    k = 10
    
    # Precompute quadrature points and weights
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    x_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    # x_theta is cos(theta), w_theta are weights
    # dOmega = sin(theta) dtheta dphi, and d(cos(theta)) = -sin(theta) dtheta
    # So integral over sphere = integral_{-1}^{1} integral_0^{2pi} f(cos(theta), phi) d(cos(theta)) dphi
    # With Gauss-Legendre: sum_i w_i f(x_i, phi_j) * (2pi/n_phi) ... actually
    # The integral is: sum_i sum_j w_i * (2pi/n_phi) * f(x_i, phi_j)
    # But we need to be careful: the surface measure dOmega = sin(theta)dtheta dphi
    # Let u = cos(theta), then du = -sin(theta) dtheta, so dOmega = du dphi
    # Integral = integral_{u=-1}^{1} integral_{phi=0}^{2pi} f(u, phi) du dphi
    # Gauss-Legendre approximates integral_{-1}^{1} g(u) du = sum_i w_i g(x_i)
    # So: integral ≈ sum_i w_i * (1/n_phi) * sum_j f(x_i, phi_j) * (2pi/n_phi)
    # Actually: integral_0^{2pi} h(phi) dphi ≈ (2pi/n_phi) sum_j h(phi_j)
    # So: integral ≈ sum_i w_i * (2pi/n_phi) * sum_j f(x_i, phi_j)
    # But this double counts. Let me redo:
    # I = integral_{-1}^{1} integral_0^{2pi} f(u, phi) du dphi
    #   ≈ sum_i w_i * [integral_0^{2pi} f(x_i, phi) dphi]
    #   ≈ sum_i w_i * (2pi/n_phi) * sum_j f(x_i, phi_j)
    
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, x_theta, indexing='ij')
    # theta_grid is cos(theta), shape (n_theta, n_phi)
    # Let's make it (n_phi, n_theta) or just flatten
    u = x_theta  # cos(theta), shape (n_theta,)
    phi_vals = phi  # shape (n_phi,)
    
    # Create full grid
    u_grid, phi_grid = np.meshgrid(u, phi_vals, indexing='ij')
    # u_grid shape: (n_theta, n_phi)
    
    # Quadrature weights: w_i * (2pi/n_phi) for each (i,j) pair
    w_phi = 2*np.pi / n_phi
    weights = np.outer(w_theta, np.ones(n_phi)) * w_phi
    weights_flat = weights.ravel()  # shape (n_theta * n_phi,)
    
    # Precompute the spherical harmonic basis values at all quadrature points
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    def N_coeff(m):
        m_abs = abs(m)
        return np.sqrt(21.0/(4*np.pi) * float(factorial(k - m_abs, exact=True)) / float(factorial(k + m_abs, exact=True)))
    
    # Build basis matrix: B[point_idx, m_idx] = Y_{m_idx}(u, phi)
    # m_idx goes from 0 to 20, corresponding to m = -10, -9, ..., 0, ..., 9, 10
    n_points = n_theta * n_phi
    B = np.zeros((n_points, 2*k+1), dtype=np.float64)
    
    for m in range(-k, k+1):
        m_idx = m + k  # index in the coefficient vector
        m_abs = abs(m)
        
        # Associated Legendre function with Condon-Shortley phase
        # scipy.special.lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley phase
        P = lpmv(m_abs, k, u_grid)  # shape (n_theta, n_phi)
        
        N_m = N_coeff(m)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * phi_grid)
        else:  # m < 0
            Y = np.sqrt(2.0) * N_m * P * np.sin(m_abs * phi_grid)
        
        B[:, m_idx] = Y.ravel()
    
    # Now f = B @ c, where c is the coefficient vector
    # ||f||_2^2 = sum_i w_i f_i^2 = (B@c)^T W (B@c) where W = diag(weights_flat)
    # ||f||_4^2 = sum_i w_i f_i^4 ... wait, ||f||_4 = (integral |f|^4 dOmega)^{1/4}
    # So ||f||_4^4 = sum_i w_i f_i^4
    # And ||f||_2^2 = sum_i w_i f_i^2
    # R = ||f||_4 / ||f||_2 = (sum w_i f_i^4)^{1/4} / (sum w_i f_i^2)^{1/2}
    
    # Let's define the objective more carefully.
    # R(c) = (sum_i w_i f_i^4)^{1/4} / (sum_i w_i f_i^2)^{1/2}
    # We want to minimize R(c).
    
    # Since R is scale-invariant (R(alpha*c) = R(c) for alpha > 0), we can add a constraint
    # like ||c||_2 = 1 or ||f||_2 = 1 to make the optimization well-posed.
    
    # Let's use the constraint ||f||_2^2 = sum_i w_i f_i^2 = 1
    # Then R = (sum_i w_i f_i^4)^{1/4}
    # Minimize R subject to sum_i w_i f_i^2 = 1
    
    # Alternatively, minimize R^4 = sum_i w_i f_i^4 subject to sum_i f_i^2 w_i = 1
    # Or even simpler: minimize (sum w_i f_i^4) / (sum w_i f_i^2)^2
    # because R^4 = (sum w_i f_i^4) / (sum w_i f_i^2)^2
    
    def objective_and_grad(c):
        f = B @ c
        f2 = f**2
        f4 = f**4
        sum_w_f2 = np.dot(weights_flat, f2)
        sum_w_f4 = np.dot(weights_flat, f4)
        
        if sum_w_f2 <= 0:
            return 0.0, np.zeros_like(c)
        
        # R^4 = sum_w_f4 / sum_w_f2^2
        val = sum_w_f4 / (sum_w_f2**2)
        
        # Gradient of val with respect to c
        # dval/dc = (1/sum_w_f2^2) * d(sum_w_f4)/dc - 2*sum_w_f4/sum_w_f2^3 * d(sum_w_f2)/dc
        # d(sum_w_f4)/dc = d/dc sum_i w_i (B@c)_i^4 = 4 * sum_i w_i (B@c)_i^3 * B_i^T
        #                  = 4 * B^T (w * f^3)
        # d(sum_w_f2)/dc = d/dc sum_i w_i (B@c)_i^2 = 2 * B^T (w * f)
        
        w_f3 = weights_flat * f**3
        w_f = weights_flat * f
        
        grad_sum_w_f4 = 4.0 * (B.T @ w_f3)
        grad_sum_w_f2 = 2.0 * (B.T @ w_f)
        
        grad = grad_sum_w_f4 / (sum_w_f2**2) - 2.0 * sum_w_f4 / (sum_w_f2**3) * grad_sum_w_f2
        
        return val, grad
    
    # Use L-BFGS-B with multiple random initializations
    best_val = np.inf
    best_c = np.ones(2*k+1, dtype=np.float64)
    
    n_restarts = 50
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        # Generate random initial point
        c0 = rng.standard_normal(2*k+1)
        # Normalize to unit L2 norm of f
        f0 = B @ c0
        sum_w_f2 = np.dot(weights_flat, f0**2)
        if sum_w_f2 > 0:
            c0 = c0 / np.sqrt(sum_w_f2)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective_and_grad,
                c0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.success or result.fun < best_val:
                val, _ = objective_and_grad(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
        except Exception:
            pass
    
    # Additional refinement with more iterations from the best point found
    if time.time() < deadline - 60:
        try:
            result = minimize(
                objective_and_grad,
                best_c,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if result.fun < best_val:
                val, _ = objective_and_grad(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
        except Exception:
            pass
    
    # Normalize the result to have reasonable magnitude
    # The problem says max|c| should be between 1e-3 and 1e5
    f_best = B @ best_c
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1.0 / max_abs_c)
    
    # Final check
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(2*k+1, dtype=np.float64)
    
    return best_c


# EVOLVE-BLOCK-END
