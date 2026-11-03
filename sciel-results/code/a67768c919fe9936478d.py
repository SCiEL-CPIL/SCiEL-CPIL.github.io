"""Optimized search for Hermite coefficients minimizing the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 180.0  # Safety margin before 210s kill

def _hermite_poly(n, x):
    """Compute physicists' Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    Hn_minus_2 = np.ones_like(x)
    Hn_minus_1 = 2 * x
    for k in range(2, n + 1):
        Hn = 2 * x * Hn_minus_1 - 2 * (k - 1) * Hn_minus_2
        Hn_minus_2 = Hn_minus_1
        Hn_minus_1 = Hn
    return Hn_minus_1

def _compute_c12(c0, c4, c8):
    """Compute c12 such that g(0) = 0."""
    # H0(0)=1, H4(0)=-6, H8(0)=105, H12(0)=-945
    # c0*1 + c4*(-6) + c8*105 + c12*(-945) = 0
    return (c0 - 6*c4 + 105*c8) / 945.0

def _uncertainty_product(c0, c4, c8):
    """Compute the uncertainty product A(g)*A(F(g)) for the Hermite-Gaussian function."""
    c12 = _compute_c12(c0, c4, c8)
    
    # Use numerical quadrature to compute the necessary norms
    # g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/4)
    # 
    # For the uncertainty inequality, we need:
    # A(g) related to ||g||_2 and ||x*g||_2 (or similar)
    # A(F(g)) related to ||F(g)||_2 and ||xi*F(g)||_2
    #
    # By Parseval, ||F(g)||_2 = ||g||_2 (up to normalization)
    # And ||xi*F(g)||_2 = ||d/dx g||_2 (up to normalization)
    #
    # So the uncertainty product is proportional to (||g||_2^2 * ||g'||_2^2)
    # or similar combinations.
    
    # Let's compute numerically
    x = np.linspace(-20, 20, 100000)
    dx = x[1] - x[0]
    
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    gauss = np.exp(-x**2 / 4)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # Compute ||g||_2^2
    norm_g_sq = np.sum(g**2) * dx
    
    # Compute g'(x) analytically
    # d/dx [Hn(x) * exp(-x^2/4)] = Hn'(x) * exp(-x^2/4) + Hn(x) * (-x/2) * exp(-x^2/4)
    # Hn'(x) = 2n * H_{n-1}(x)
    H0p = np.zeros_like(x)  # H0' = 0
    H4p = 2 * 4 * _hermite_poly(3, x)  # H4' = 8*H3
    H8p = 2 * 8 * _hermite_poly(7, x)  # H8' = 16*H7
    H12p = 2 * 12 * _hermite_poly(11, x)  # H12' = 24*H11
    
    gp = (c0 * H0p + c4 * H4p + c8 * H8p + c12 * H12p) * gauss - (x / 2) * g
    
    # Compute ||g'||_2^2
    norm_gp_sq = np.sum(gp**2) * dx
    
    # The uncertainty product is typically ||g||_2 * ||g'||_2 or similar
    # For the standard Heisenberg-type inequality, it's (∫|g|^2)(∫x^2|g|^2) >= (1/4)(∫|g|^2)^2
    # But here we're looking at a different form. Let's use the product of L2 norms.
    
    if norm_g_sq <= 0:
        return 1e10
    
    # The C4 upper bound is likely related to the ratio or product of these quantities
    # Let's use the standard uncertainty product: sqrt(∫|g|^2 * ∫x^2|g|^2) * sqrt(∫|F(g)|^2 * ∫xi^2|F(g)|^2)
    # By Parseval and properties of Fourier transform:
    # ∫xi^2|F(g)|^2 = ∫|g'|^2 (up to constants)
    
    # Compute ∫x^2|g|^2
    x2_g_sq = np.sum(x**2 * g**2) * dx
    
    # The uncertainty product is sqrt(∫|g|^2 * ∫x^2|g|^2) * sqrt(∫|g|^2 * ∫|g'|^2)
    # = (∫|g|^2) * sqrt(∫x^2|g|^2 * ∫|g'|^2)
    
    if x2_g_sq <= 0 or norm_gp_sq <= 0:
        return 1e10
    
    uncertainty_product = norm_g_sq * np.sqrt(x2_g_sq * norm_gp_sq)
    
    return uncertainty_product

def construct():
    """Return three coefficients minimizing the C4 upper bound."""
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    
    if remaining < 10:
        # Not enough time, return seed values
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # First, try a quick local optimization from the seed point
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        return _uncertainty_product(c0, c4, c8)
    
    best_result = None
    best_val = objective(seed)
    
    # Try local optimization with BFGS
    try:
        res = minimize(objective, seed, method='BFGS', 
                      options={'maxiter': 500, 'gtol': 1e-10})
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception:
        pass
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    
    # If we have time, try differential evolution for global search
    if remaining > 60:
        try:
            # Use differential evolution with a smaller population for speed
            def de_objective(coeffs):
                c0, c4, c8 = coeffs
                return _uncertainty_product(c0, c4, c8)
            
            bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
            de_res = differential_evolution(de_objective, bounds, 
                                           maxiter=50, popsize=15,
                                           seed=42, tol=1e-8)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_result = de_res.x
        except Exception:
            pass
    
    if best_result is not None:
        return np.array(best_result)
    return seed
# EVOLVE-BLOCK-END
