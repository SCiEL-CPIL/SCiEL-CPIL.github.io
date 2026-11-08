"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad, quad_vec

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 175.0  # seconds, with safety margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    # H_k(0) for even k: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-103950
    h0_0 = 1.0
    h4_0 = -12.0
    h8_0 = 1680.0
    h12_0 = -103950.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12

def _g_function(x, c0, c4, c8, c12):
    """Compute g(x) = sum c_k H_k(x) exp(-x^2/2)"""
    H0 = np.ones_like(x) if np.ndim(x) > 0 else 1.0
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 76800*x**10 + 518400*x**8 - 1548480*x**6 + 2342400*x**4 - 1548480*x**2 + 103950
    
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    return g * np.exp(-x**2 / 2.0)

def _compute_bound_analytic(c0, c4, c8):
    """Compute the upper bound using adaptive quadrature for accuracy."""
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Define the function for quadrature
    def abs_g(x):
        val = _g_function(np.array([x]), c0, c4, c8, c12)
        return np.abs(val[0])
    
    # Integrate |g(x)| from -inf to inf using adaptive quadrature
    # Split into intervals for better convergence
    A_f, err_f = quad(abs_g, -15, 15, limit=200, epsabs=1e-12, epsrel=1e-10)
    A_f += quad(abs_g, -np.inf, -15, limit=50, epsabs=1e-12, epsrel=1e-10)[0]
    A_f += quad(abs_g, 15, np.inf, limit=50, epsabs=1e-12, epsrel=1e-10)[0]
    
    # For the Fourier transform: with convention F(f)(xi) = integral f(x) e^{-i x xi} dx
    # For H_k(x) e^{-x^2/2}, the FT is sqrt(2*pi) * (-i)^k H_k(xi) e^{-xi^2/2}
    # For k=0,4,8,12: (-i)^k = 1, so F(f)(xi) = sqrt(2*pi) * g(xi)
    # Thus A_F = sqrt(2*pi) * A_f
    # So A(f)*A(F(f)) = sqrt(2*pi) * A_f^2
    
    A_F = np.sqrt(2 * np.pi) * A_f
    return A_f * A_F

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound_analytic(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_analytic,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try multi-start with different initial points
    if _time_left() > 50:
        try:
            # Try several random starting points
            np.random.seed(42)
            for _ in range(5):
                if _time_left() < 20:
                    break
                start = np.random.uniform([-5, 5, -0.5, 0.5], [5, 5, 0.5, -0.5])[:3]
                res = minimize(
                    _compute_bound_analytic,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 80:
        try:
            # Reasonable bounds for coefficients
            bounds = [(-10, 10), (-10, 10), (-1, 1)]
            res_de = differential_evolution(
                _compute_bound_analytic,
                bounds,
                maxiter=100,
                popsize=12,
                seed=42,
                tol=1e-8,
                polish=True
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Final high-precision polish pass
    if _time_left() > 15 and not np.array_equal(best_params, seed):
        try:
            res2 = minimize(
                _compute_bound_analytic,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
