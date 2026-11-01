import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 210.0
SAFETY_MARGIN = 10.0

def _hermite_physicist(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    # Use recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    # But for specific small n, we can hardcode or use a general recurrence.
    # Given we only need 0, 4, 8, 12, we can compute up to 12.
    # For vectorized x:
    h0 = np.ones_like(x)
    h1 = 2 * x
    if n == 0:
        return h0
    if n == 1:
        return h1
    h_prev = h0
    h_curr = h1
    for k in range(1, n):
        h_next = 2 * x * h_curr - 2 * k * h_prev
        h_prev = h_curr
        h_curr = h_next
    return h_curr

def _get_coefficients(c0, c4, c8):
    """
    Calculate c12 such that g(0) = 0.
    g(x) = e^{-pi x^2} (c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x))
    g(0) = c0 H0(0) + c4 H4(0) + c8 H8(0) + c12 H12(0) = 0
    """
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 10395000
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 10395000.0
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c12

def _objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = _get_coefficients(c0, c4, c8)
    
    # Define the polynomial part P(x)
    # We need to integrate |e^{-pi x^2} P(x)| and |e^{-pi x^2} Q(x)|
    # where Q(x) is the polynomial for the Fourier transform.
    # FT of e^{-pi x^2} H_k(x) is (-i)^k e^{-pi x^2} H_k(x).
    # So Q(x) = c0 H0(x) + c4 (-i)^4 H4(x) + c8 (-i)^8 H8(x) + c12 (-i)^12 H12(x)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    # So Q(x) = P(x). The L1 norms are identical!
    # ||g||_1 == ||hat{g}||_1.
    # Objective: minimize ||g||_1^2 / ||g||_2^2 ? Or just ||g||_1 if ||g||_2=1?
    # Let's assume we minimize ||g||_1 subject to ||g||_2 = 1.
    # This is equivalent to minimizing ||g||_1^2 / ||g||_2^2.
    
    def integrand_abs(x):
        # Calculate P(x)
        # Use vectorized Hermite
        h0 = _hermite_physicist(0, x)
        h4 = _hermite_physicist(4, x)
        h8 = _hermite_physicist(8, x)
        h12 = _hermite_physicist(12, x)
        
        p_val = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
        g_val = np.exp(-np.pi * x**2) * p_val
        return np.abs(g_val)
        
    def integrand_sq(x):
        h0 = _hermite_physicist(0, x)
        h4 = _hermite_physicist(4, x)
        h8 = _hermite_physicist(8, x)
        h12 = _hermite_physicist(12, x)
        
        p_val = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
        g_val = np.exp(-np.pi * x**2) * p_val
        return g_val**2

    # Integrate from -L to L. Choose L large enough.
    # e^{-pi x^2} decays fast. L=10 is very safe.
    L = 10.0
    
    # Use quad for accuracy
    l1_norm, _ = integrate.quad(integrand_abs, -L, L, limit=100)
    l2_norm_sq, _ = integrate.quad(integrand_sq, -L, L, limit=100)
    
    if l2_norm_sq < 1e-12:
        return 1e10
        
    # Minimize L1^2 / L2^2 is equivalent to minimizing L1 / L2.
    # The "upper bound" score is likely this ratio.
    return (l1_norm * l1_norm) / l2_norm_sq

def construct():
    global START_TIME, TIME_LIMIT, SAFETY_MARGIN
    
    # Initial guess from seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = _objective(c_seed)
    best_coeffs = c_seed.copy()
    
    # Optimization
    # Use L-BFGS-B or Powell. Powell is robust for non-smooth (abs) but slow.
    # The abs function makes it non-differentiable at roots of P(x).
    # However, the roots are isolated. We can try using a smooth approximation or just rely on the optimizer's robustness.
    # Alternatively, minimize the smooth part? No, L1 is key.
    
    # Let's try a few restarts or a single robust run.
    # Given the time limit, a single careful run is safer.
    
    try:
        # Use Powell method as it doesn't require gradients and handles non-smoothness better than L-BFGS.
        # Set maxiter to a reasonable number.
        res = opt.minimize(
            _objective, 
            c_seed, 
            method='Powell', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
            
    except Exception as e:
        pass
        
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return best_coeffs
        
    # Try a second phase with L-BFGS-B if we have time, starting from the Powell result
    # L-BFGS-B might get stuck in local minima due to non-smoothness, but can polish.
    # Actually, L-BFGS-B assumes smoothness. The abs function is not smooth at zeros.
    # We can replace abs(x) with sqrt(x^2 + eps) for smoothness during optimization.
    
    def _objective_smooth(coeffs, eps=1e-6):
        c0, c4, c8 = coeffs
        c12 = _get_coefficients(c0, c4, c8)
        
        def integrand_abs(x):
            h0 = _hermite_physicist(0, x)
            h4 = _hermite_physicist(4, x)
            h8 = _hermite_physicist(8, x)
            h12 = _hermite_physicist(12, x)
            p_val = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
            g_val = np.exp(-np.pi * x**2) * p_val
            return np.sqrt(g_val**2 + eps)
            
        def integrand_sq(x):
            h0 = _hermite_physicist(0, x)
            h4 = _hermite_physicist(4, x)
            h8 = _hermite_physicist(8, x)
            h12 = _hermite_physicist(12, x)
            p_val = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
            g_val = np.exp(-np.pi * x**2) * p_val
            return g_val**2

        L = 10.0
        l1_norm, _ = integrate.quad(integrand_abs, -L, L, limit=100)
        l2_norm_sq, _ = integrate.quad(integrand_sq, -L, L, limit=100)
        
        if l2_norm_sq < 1e-12:
            return 1e10
        return (l1_norm * l1_norm) / l2_norm_sq

    try:
        res2 = opt.minimize(
            _objective_smooth, 
            best_coeffs, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        # Verify with the true objective
        true_score = _objective(res2.x)
        if true_score < best_score:
            best_score = true_score
            best_coeffs = res2.x
            
    except Exception as e:
        pass

    return best_coeffs

# EVOLVE-BLOCK-END
