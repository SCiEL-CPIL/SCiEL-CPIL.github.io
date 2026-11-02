"""
Global optimization of Hermite coefficients for the uncertainty bound.
Uses differential evolution for global search followed by local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature with N=30 (exact for degree up to 58)
    nodes, weights = hermgauss(30)
    sqrt2 = np.sqrt(2.0)
    x_nodes = sqrt2 * nodes

    # Precompute Hermite polynomial values at quadrature nodes
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x8 * x4

    H0_vals = np.ones_like(x_nodes)
    H4_vals = 16.0 * x4 - 48.0 * x2 + 12.0
    H8_vals = (256.0 * x8 - 2304.0 * x6 + 5760.0 * x4
              - 4320.0 * x2 + 504.0)
    H12_vals = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
                - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2
                + 10395.0)

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def objective(coeffs):
        """Compute the upper bound on C4."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        P_vals = a0 * H0_vals + a4 * H4_vals + a8 * H8_vals + a12 * H12_vals
        Q_vals = a0 * H0_vals + a4 * H4_vals + a8 * H8_vals - a12 * H12_vals
        
        norm2_g_sq = sqrt2 * np.sum(weights * P_vals**2)
        norm2_q_sq = sqrt2 * np.sum(weights * Q_vals**2)
        norm4_g_4 = sqrt2 * np.sum(weights * P_vals**4)
        norm4_q_4 = sqrt2 * np.sum(weights * Q_vals**4)
        
        if (norm2_g_sq < 1e-15 or norm2_q_sq < 1e-15 or 
            norm4_g_4 < 1e-15 or norm4_q_4 < 1e-15):
            return 1e10
            
        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25
        
        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # Phase 1: Differential evolution for global search
    best_coeffs = seed.copy()
    best_val = objective(best_coeffs)
    
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import differential_evolution
            
            # Define bounds for coefficients
            bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
            
            # Run differential evolution
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                mut=(0.5, 1.0),
                recomb=0.8,
                polish=True,
                init='latinhypercube'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass

    # Phase 3: Final polish with Powell
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
