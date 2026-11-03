"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
import time

# Precompute Gauss-Hermite nodes and weights
# Need to integrate polynomials up to degree 24 (x^2 * P(x)^2 where P is degree 12)
# N=16 is safe (integrates exactly up to degree 31)
N_QUAD = 16
_x_nodes, _w_weights = hermgauss(N_QUAD)

# Precompute Hermite polynomial values at nodes
_H0 = np.ones(N_QUAD)
_H4 = eval_hermite(4, _x_nodes)
_H8 = eval_hermite(8, _x_nodes)
_H12 = eval_hermite(12, _x_nodes)

# Values at x=0 for constraint
_h0_0 = 1.0
_h4_0 = -3.0  # H4(0) = -3 for physicist's Hermite
_h8_0 = 105.0
_h12_0 = -945.0  # H12(0) = -945


def _get_c12(c0, c4, c8):
    """Compute c12 such that g(0) = 0."""
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    return -(c0 * _h0_0 + c4 * _h4_0 + c8 * _h8_0) / _h12_0


def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 for given coefficients using Gauss-Hermite quadrature."""
    c12 = _get_c12(c0, c4, c8)
    
    # Evaluate P(x) at Gauss-Hermite nodes
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    P = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    
    # g(x) = P(x) * exp(-x^2/2)
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    
    # Integrate |g|^2 dx = integral P(x)^2 * exp(-x^2) dx
    # Using Gauss-Hermite: sum w_i * f(x_i) where f includes exp(-x^2)
    # But hermgauss integrates against exp(-x^2), so:
    # integral f(x) exp(-x^2) dx = sum w_i f(x_i)
    # Here f(x) = P(x)^2, so:
    norm_sq = np.sum(_w_weights * P**2)
    
    if norm_sq < 1e-15:
        return 1e10
    
    # Second moment in x: integral x^2 |g|^2 dx / integral |g|^2 dx
    x_sq = _x_nodes**2
    second_moment_x = np.sum(_w_weights * x_sq * P**2) / norm_sq
    
    # For Fourier transform: since all Hermite degrees are even,
    # ĝ(ξ) = P(ξ) * exp(-ξ^2/2) (same functional form)
    # So the second moment in frequency space is identical:
    # integral ξ^2 |ĝ|^2 dξ / integral |ĝ|^2 dξ = second_moment_x
    
    # The uncertainty product bound C4 = second_moment_x * second_moment_x = second_moment_x^2
    C4 = second_moment_x ** 2
    
    return C4


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s kill
    
    # Initial guess from seed
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    best_c = np.array([c0_init, c4_init, c8_init])
    best_val = _compute_bound(best_c[0], best_c[1], best_c[2])
    
    # Optimization objective
    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    # Try Nelder-Mead optimization
    try:
        res = minimize(objective, best_c, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                              'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    # Try additional local optimizations from perturbed starting points
    rng = np.random.default_rng(42)
    for i in range(10):
        if time.time() > deadline - 5:
            break
        c_start = best_c + rng.normal(0, 0.01, size=3)
        try:
            res = minimize(objective, c_start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)

# EVOLVE-BLOCK-END
