import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the normalized uncertainty bound, with H12 determined by g(0)=0.
    """
    # Hermite polynomial values at 0 for n = 0, 4, 8, 12
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def get_all_coeffs(a0, a4, a8):
        """Return all four coefficients with a12 determined by g(0)=0."""
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        return np.array([a0, a4, a8, a12])
    
    def g_func(x, coeffs):
        """Evaluate g(x) = sum a_n H_n(x) exp(-x^2/2)"""
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            result += a * eval_hermite(n, x) * np.exp(-x**2 / 2)
        return result
        
    def g_hat_func(xi, coeffs):
        """
        Fourier transform of g. Using convention where
        F[H_n(x) e^{-x^2/2}](xi) = (i)^n H_n(xi) e^{-xi^2/2}
        """
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(xi) if isinstance(xi, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            result += a * (1j)**n * eval_hermite(n, xi) * np.exp(-xi**2 / 2)
        return result
        
    def compute_norms(coeffs):
        """Compute ||g||_1, ||g_hat||_1, and ||g||_2 via numerical quadrature."""
        def abs_g(x):
            return np.abs(g_func(x, coeffs))
        
        def abs_g_hat(xi):
            return np.abs(g_hat_func(xi, coeffs))
            
        def g_sq(x):
            val = g_func(x, coeffs)
            return np.abs(val)**2

        l1_g, _ = quad(lambda x: float(abs_g(x)), -10, 10, limit=200)
        l1_g_hat, _ = quad(lambda xi: float(abs_g_hat(xi)), -10, 10, limit=200)
        
        # L2 norm squared is integral of |g|^2
        l2_sq_g, _ = quad(lambda x: float(g_sq(x)), -10, 10, limit=200)
        
        return l1_g, l1_g_hat, l2_sq_g

    def objective(params):
        """Objective: minimize the normalized uncertainty product."""
        a0, a4, a8 = params
        coeffs = get_all_coeffs(a0, a4, a8)
        
        # Check if coefficients are effectively zero
        norm_coeffs = np.linalg.norm(coeffs)
        if norm_coeffs < 1e-12:
            return 1e6
            
        # Normalize to unit L2 norm to handle scale invariance
        # The objective is (L1_g * L1_g_hat) / L2_sq_g
        # If we normalize coeffs to have L2 norm 1, then L2_sq_g = 1
        # So we just minimize L1_g * L1_g_hat
        coeffs_norm = coeffs / norm_coeffs
        
        l1_g, l1_g_hat, l2_sq_g = compute_norms(coeffs_norm)
        
        if l2_sq_g < 1e-12:
            return 1e6
            
        # Since coeffs are normalized to L2=1, l2_sq_g should be 1.
        # We return the product L1_g * L1_g_hat
        return l1_g * l1_g_hat

    # Multi-start optimization
    best_result = None
    best_fun = float('inf')
    
    # Define a set of initial guesses
    # We can use random points on the sphere or a grid
    rng = np.random.default_rng(42)
    
    # Try a few random starts
    for i in range(20):
        # Random start in a box
        start = rng.uniform(-1, 1, size=3)
        
        try:
            res = minimize(objective, start, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res.x
        except Exception:
            continue
            
    if best_result is None:
        # Fallback
        best_result = np.array([-1.0, 0.1, 0.01])
        
    return np.array(best_result)
