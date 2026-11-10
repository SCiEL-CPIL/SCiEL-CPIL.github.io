import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    k = 10
    n_m = 2 * k + 1  # 21 coefficients
    
    # 1. Setup Quadrature
    # 70-point Gauss-Legendre in cos(theta)
    mu, w_mu = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Meshgrid
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = MU.flatten()
    phi_flat = PHI.flatten()
    
    # Weights: w_i = w_mu[i] * d_phi
    # The integral over sphere is sum w_i f_i^p
    # Total mass should be 4*pi
    # sum(w_mu) = 2. sum over phi is n_phi * d_phi = 2*pi.
    # So sum(weights) = 2 * 2*pi = 4*pi. Correct.
    weights = np.repeat(w_mu, n_phi)
    
    # 2. Precompute Spherical Harmonics Y
    # Y shape: (N, 21)
    # m = -10 .. 10
    Y = np.zeros((mu_flat.size, n_m), dtype=np.float64)
    
    # Normalization constants N_m
    # N_m = sqrt(21/(4*pi) * (10-m)! / (10+m)!)
    # Use log-gamma to avoid overflow/underflow if needed, but 10! is small.
    import math
    N = np.zeros(n_m)
    for i in range(n_m):
        m = i - k
        abs_m = abs(m)
        # (10-abs_m)! / (10+abs_m)!
        # 10! is 3628800. 20! is large but fits in float64?
        # 20! ~ 2.4e18. 10! ~ 3.6e6. Ratio is small.
        # math.factorial is exact integer.
        num = math.factorial(k - abs_m)
        den = math.factorial(k + abs_m)
        N[i] = np.sqrt((21.0 / (4.0 * np.pi)) * (num / den))
    
    # Compute P_10^m for all m
    # lpmv(m, l, x) computes P_l^m(x) with Condon-Shortley phase.
    for i in range(n_m):
        m = i - k
        abs_m = abs(m)
        P = lpmv(abs_m, k, mu_flat)
        
        if m == 0:
            Y[:, i] = N[i] * P
        elif m > 0:
            Y[:, i] = np.sqrt(2.0) * N[i] * P * np.cos(m * phi_flat)
        else:
            Y[:, i] = np.sqrt(2.0) * N[i] * P * np.sin(abs_m * phi_flat)
            
    # 3. Define Objective and Gradient
    # Minimize R = ||f||_4 / ||f||_2
    # Let J = R^2 = ||f||_4^2 / ||f||_2^2 = S4 / S2
    # S2 = sum(w * f^2)
    # S4 = sum(w * f^4)
    # grad J = (1/S2) * dS4/dc - (S4/S2^2) * dS2/dc
    # dS2/dc = 2 * Y^T (w * f)
    # dS4/dc = 4 * Y^T (w * f^3)
    
    def objective_and_grad(c):
        f = Y.dot(c)
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        
        S2 = np.dot(weights, f2)
        S4 = np.dot(weights, f4)
        
        # Avoid division by zero (though f=0 is not a minimum for R, it's a singular point)
        if S2 < 1e-12:
            return 1e12, np.zeros_like(c)
            
        J = S4 / S2
        
        # Gradients
        # term1 = 4/S2 * Y^T (w * f^3)
        # term2 = 2*S4/S2^2 * Y^T (w * f)
        
        # w_f3 = weights * f^3
        # w_f = weights * f
        
        # We can compute Y^T (w * v) as Y.T @ (w * v)
        # To save memory, we can do:
        # grad = (4/S2) * (Y.T @ (weights * f3)) - (2*S4/S2**2) * (Y.T @ (weights * f))
        
        # Optimization: 
        # Y.T @ (w * v) is equivalent to (Y * w[:, None]).T @ v
        # Let Yw = Y * weights[:, None]
        # Then grad = 4/S2 * Yw.T @ f3 - 2*S4/S2^2 * Yw.T @ f
        
        # Precompute Yw outside?
        # Yw is 9100 x 21. Small.
        
        Yw_T_f = Yw.T.dot(f)
        Yw_T_f3 = Yw.T.dot(f3)
        
        grad = (4.0 / S2) * Yw_T_f3 - (2.0 * S4 / (S2 * S2)) * Yw_T_f
        
        return J, grad

    # Precompute Yw.T for efficiency
    Yw = Y * weights[:, np.newaxis]
    Yw_T = Yw.T
    
    def objective_and_grad_fast(c):
        f = Y.dot(c)
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        
        S2 = np.dot(weights, f2)
        S4 = np.dot(weights, f4)
        
        if S2 < 1e-12:
            return 1e12, np.zeros_like(c)
            
        J = S4 / S2
        
        Yw_T_f = Yw_T.dot(f)
        Yw_T_f3 = Yw_T.dot(f3)
        
        grad = (4.0 / S2) * Yw_T_f3 - (2.0 * S4 / (S2 * S2)) * Yw_T_f
        
        return J, grad

    # 4. Optimization Loop
    start_time = time.time()
    time_limit = 600.0  # Safety margin
    
    best_c = np.ones(n_m)
    best_val = objective_and_grad_fast(best_c)[0]
    
    # Initial points
    # 1. All ones
    # 2. Random normal
    # 3. Sign alternating
    # 4. High m emphasis
    
    rng = np.random.default_rng(42)
    
    starts = [
        np.ones(n_m),
        rng.normal(size=n_m),
        np.array([(-1)**(i-k) for i in range(n_m)]),
        np.zeros(n_m),
    ]
    
    # Add some random starts
    for _ in range(5):
        starts.append(rng.normal(size=n_m))
    
    for c0 in starts:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective_and_grad_fast, c0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final check for validity
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_m)
        
    # Scale doesn't matter for R, but ensure it's not zero
    if np.max(np.abs(best_c)) < 1e-3:
        best_c = np.ones(n_m)
    elif np.max(np.abs(best_c)) > 1e5:
        best_c = best_c / np.max(np.abs(best_c)) * 1.0

    return best_c
