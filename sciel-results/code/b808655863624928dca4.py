"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 180.0  # seconds, with safety margin before 210s kill

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

def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F(f)) for the given coefficients."""
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Use a fine grid to compute L1 norm numerically
    x = np.linspace(-20, 20, 200001)
    dx = x[1] - x[0]
    
    # g(x) = sum c_k H_k(x) exp(-x^2/2)
    # H_k(x) for k = 0, 4, 8, 12
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 76800*x**10 + 518400*x**8 - 1548480*x**6 + 2342400*x**4 - 1548480*x**2 + 103950
    
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = g * np.exp(-x**2 / 2.0)
    
    # L1 norm of g
    A_f = np.sum(np.abs(g)) * dx
    
    # Fourier transform: F{H_k(x) e^{-x^2/2}} = (-i)^k H_k(x) e^{-x^2/2} (up to normalization)
    # With standard FT convention: F(f)(xi) = integral f(x) e^{-2 pi i x xi} dx
    # For Hermite-Gaussian with e^{-x^2/2}, the FT is a phase rotation: (-i)^k for H_k
    # The magnitude |F(f)(xi)| = |sum c_k (-i)^k H_k(xi) e^{-xi^2/2}|
    # Since (-i)^k for k=0,4,8,12: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1
    # So all phases are +1! The FT has the same form.
    # Thus A_F = A_f (by symmetry)
    # Actually let me verify: (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1. Yes.
    # (-i)^8 = 1, (-i)^12 = 1.
    # So |F(f)(xi)| = |g(xi)|, meaning A_F = A_f.
    
    # But wait - the normalization of the FT matters. Let's use the unitary FT or standard.
    # For the uncertainty principle, typically we use:
    # A(f) = integral |f(x)| dx
    # A(F(f)) = integral |F(f)(xi)| dxi
    # where F(f)(xi) = integral f(x) e^{-2 pi i x xi} dx
    # 
    # For f(x) = H_k(x) e^{-x^2/2}, F(f)(xi) = (-i)^k H_k(xi) e^{-xi^2/2} * sqrt(2*pi) * (normalization)
    # Actually, the exact normalization depends on convention. Let me just compute numerically.
    
    # Let's compute the FT numerically to be safe
    # Use FFT on the grid
    N = len(x)
    xi = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    F_g = np.fft.fft(g) * dx  # FT approximation
    # Reorder to be centered
    F_g = np.fft.fftshift(F_g)
    xi = np.fft.fftshift(xi)
    
    # L1 norm of FT
    A_F = np.sum(np.abs(F_g)) * (xi[1] - xi[0]) if len(xi) > 1 else 0.0
    
    return A_f * A_F

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 60:
        try:
            # Reasonable bounds for coefficients
            bounds = [(-10, 10), (-10, 10), (-1, 1)]
            res_de = differential_evolution(
                _compute_bound,
                bounds,
                maxiter=200,
                popsize=15,
                seed=42,
                tol=1e-8
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Refine with Nelder-Mead from best found
    if _time_left() > 20 and not np.array_equal(best_params, seed):
        try:
            res2 = minimize(
                _compute_bound,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
