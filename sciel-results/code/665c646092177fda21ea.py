"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import time
from math import factorial, comb

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive values (the double roots).
    """
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    def laguerre_norm_sq(roots):
        """
        Compute ∫_0^∞ [prod_{i=1}^{12} (x - r_i)^2]^2 e^{-x} dx
        = ∫_0^∞ [prod_{i=1}^{12} (x - r_i)^4] e^{-x} dx
        """
        def integrand(x):
            val = 1.0
            for r in roots:
                val *= (x - r)**4
            return val * np.exp(-x)
        
        # Integrate numerically
        result, err = quad(integrand, 0, np.inf, limit=200)
        return result
    
    # This is expensive. Let's use a different approach.
    # Expand the polynomial and use the Laguerre moment formula.
    # ∫_0^∞ x^n e^{-x} dx = n!
    # So if P(x) = sum a_j x^j, then ||P||^2 = sum_{j,k} a_j a_k (j+k)!
    
    def build_polynomial(roots):
        """Build P(x) = prod (x - r_i)^2 in monomial basis (ascending order)."""
        poly = np.array([1.0])
        for r in roots:
            # (x - r)^2 = x^2 - 2*r*x + r^2
            factor = np.array([r**2, -2*r, 1.0])
            poly = np.convolve(poly, factor)
        return poly
    
    def norm_sq_from_monomial(poly):
        """Compute ||P||^2 = sum_{j,k} a_j a_k (j+k)!"""
        n = len(poly)
        norm_sq = 0.0
        for j in range(n):
            for k in range(n):
                norm_sq += poly[j] * poly[k] * factorial(j + k)
        return norm_sq
    
    def objective(roots):
        poly = build_polynomial(roots)
        return norm_sq_from_monomial(poly)
    
    # Use squared variables: r_i = s_i^2
    def objective_s(s):
        roots = np.array(s) ** 2
        return objective(roots)
    
    # Bounds: r_i in [0.01, 100], so s_i in [0.1, 10]
    s_min = 0.1
    s_max = 10.0
    bounds = [(s_min, s_max)] * 12
    
    best_score = np.inf
    best_s = None
    
    # Initial guess: roots spread out
    initial_roots = np.linspace(0.5, 30.0, 12)
    initial_s = np.sqrt(initial_roots)
    
    # Local optimization
    try:
        result = minimize(objective_s, initial_s, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 2000, 'ftol': 1e-12})
        if result.fun < best_score:
            best_score = result.fun
            best_s = result.x.copy()
    except Exception:
        pass
    
    # Differential evolution
    if time.time() < deadline - 150:
        try:
            de_result = differential_evolution(
                objective_s, bounds, seed=42, maxiter=100, popsize=15,
                tol=1e-8, polish=True, init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_s = de_result.x.copy()
        except Exception:
            pass
    
    # Final polish
    if best_s is not None and time.time() < deadline - 30:
        try:
            polish = minimize(objective_s, best_s, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 10000, 'ftol': 1e-15})
            if polish.fun < best_score:
                best_score = polish.fun
                best_s = polish.x.copy()
        except Exception:
            pass
    
    if best_s is None:
        best_s = np.sqrt(np.linspace(0.5, 30.0, 12))
    
    best_roots = np.array(best_s) ** 2
    
    # Return 12 roots (the 12 distinct double-root locations)
    return list(best_roots)
# EVOLVE-BLOCK-END
