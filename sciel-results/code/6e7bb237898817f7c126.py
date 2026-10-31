import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the normalized uncertainty bound, with H12 determined by g(0)=0.
    
    Uses multi-start local optimization (Nelder-Mead) and analytical L2 norm.
    """
    # Hermite polynomial values at 0 for n = 0, 4, 8, 12
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # Precompute L2 norm factors: ||H_n(x) e^{-x^2/2}||_2^2 = n! 2^n sqrt(pi)
    # We only need the relative weights, but let's compute the exact value for the denominator.
    # Actually, the objective is (L1_g * L1_g_hat) / L2_sq_g.
    # L2_sq_g = sum( a_n^2 * n! * 2^n * sqrt(pi) )
    
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
        
    def compute_l1_norms(coeffs):
        """Compute ||g||_1 and ||g_hat||_1 via numerical quadrature."""
        def abs_g(x):
            return np.abs(g_func(x, coeffs))
        
        def abs_g_hat(xi):
            return np.abs(g_hat_func(xi, coeffs))

        # Integration limits: The Gaussian decays fast. 10 is usually sufficient.
        # Using higher limit for safety but quad handles infinite well too.
        l1_g, _ = quad(lambda x: float(abs_g(x)), -10, 10, limit=100, epsabs=1e-10, epsrel=1e-10)
        l1_g_hat, _ = quad(lambda xi: float(abs_g_hat(xi)), -10, 10, limit=100, epsabs=1e-10, epsrel=1e-10)
        
        return l1_g, l1_g_hat

    def compute_l2_sq_norm(coeffs):
        """Compute ||g||_2^2 analytically using orthogonality."""
        n_vals = [0, 4, 8, 12]
        l2_sq = 0.0
        for n, a in zip(n_vals, coeffs):
            # Integral of (H_n(x) e^{-x^2/2})^2 dx = n! 2^n sqrt(pi)
            weight = factorial(n, exact=True) * (2**n) * np.sqrt(np.pi)
            l2_sq += (a**2) * weight
        return l2_sq

    def objective(params):
        """Objective: minimize the normalized uncertainty product."""
        a0, a4, a8 = params
        coeffs = get_all_coeffs(a0, a4, a8)
        
        # Check if coefficients are effectively zero
        if np.allclose(coeffs, 0, atol=1e-12):
            return 1e6
            
        l2_sq_g = compute_l2_sq_norm(coeffs)
        
        if l2_sq_g < 1e-12:
            return 1e6
            
        l1_g, l1_g_hat = compute_l1_norms(coeffs)
        
        # The normalized uncertainty bound is proportional to (L1_g * L1_g_hat) / L2_sq_g
        return (l1_g * l1_g_hat) / l2_sq_g

    # Multi-start optimization
    # Since the problem is scale-invariant in the objective (numerator scales as k^2, denominator as k^2),
    # the optimal point lies on a ray. We can fix the scale or just let the optimizer find a point.
    # Nelder-Mead works well for this.
    
    best_score = 1e6
    best_params = np.array([0.0, 0.0, 0.0])
    
    # Generate several random starting points
    # We sample on a sphere of radius 1 in the 3D space of (a0, a4, a8)
    # plus some random magnitudes to explore different scales (though scale doesn't matter for the value, 
    # it helps the optimizer navigate if there are numerical issues at very small/large scales).
    
    np.random.seed(42)
    num_starts = 20
    
    for i in range(num_starts):
        # Random direction on sphere
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        
        # Random magnitude (log-uniform between 0.1 and 10)
        mag = 10**np.random.uniform(-1, 1)
        
        x0 = v * mag
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 5000})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # If all failed, fallback to a known reasonable guess
    if best_score == 1e6:
        best_params = np.array([-1.0, 0.1, 0.01])
        
    return np.array(best_params)
