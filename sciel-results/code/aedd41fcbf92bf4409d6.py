"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_herm
import time

_start_time = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 10.0


def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return eval_herm(n, x)


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(Fourier(f)) for given coefficients."""
    # Derive c12 from P(0) = 0: c0 + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    H4_0 = _hermite(4, 0.0)   # = 3
    H8_0 = _hermite(8, 0.0)   # = 105
    H12_0 = _hermite(12, 0.0) # = 10395
    
    if abs(H12_0) < 1e-15:
        return np.inf
    c12 = -(c0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Build the function g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # For Fourier-invariant (self-Fourier) Hermite-Gaussian functions,
    # the bound involves L1 norms.
    # 
    # The uncertainty inequality bound C4 relates to ||g||_1 * ||g_hat||_1
    # For self-Fourier functions, g_hat = ±g, so the bound is ||g||_1^2
    # or involves the product of L1 norms of g and its Fourier transform.
    #
    # Actually, for the specific "upper bound on A(f)*A(Fourier(f))" in the
    # context of the uncertainty inequality with test functions, the bound
    # is typically computed as:
    #   C4 = sup over f of A(f) * A(Fhat(f))
    # and the test function approach gives an upper bound via:
    #   A(f) <= ||f * g||_1 * ||f||_infty ... 
    # 
    # More precisely, for the Hermite-Gaussian approach to uncertainty:
    # The bound is ||g||_1 * ||g_hat||_1 where g is the test function.
    # Since g is constructed to be Fourier-invariant (up to a sign), 
    # g_hat = e^{-i pi/2} * g (or similar), so ||g_hat||_1 = ||g||_1.
    # Thus the bound is ||g||_1^2.
    #
    # But we need to be careful: the "upper bound on C4" is what we minimize.
    # Let's compute ||g||_1 numerically.
    
    # Numerically integrate |g(x)| over the real line
    # Use a sufficiently large domain with Gaussian decay
    x = np.linspace(-20, 20, 20000)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = poly * np.exp(-x**2 / 2.0)
    
    # ||g||_1
    norm1_g = np.sum(np.abs(g)) * dx
    
    # For the Fourier transform of a Hermite-Gaussian function:
    # F{H_n(x) e^{-x^2/2}} = (-i)^n H_n(x) e^{-x^2/2} (up to normalization)
    # So |g_hat(x)| = |poly(x)| * e^{-x^2/2} = |g(x)|
    # Thus ||g_hat||_1 = ||g||_1
    norm1_ghat = norm1_g
    
    # The upper bound on C4 is the product of the L1 norms
    bound = norm1_g * norm1_ghat
    
    return bound


def _objective(coeffs):
    """Objective function for optimization."""
    c0, c4, c8 = coeffs
    try:
        val = _compute_bound(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """Find optimal coefficients for H0, H4, H8 Hermite combination."""
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed - _SAFETY_MARGIN
    if remaining < 5.0:
        # Not enough time, return prior best
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(*best_coeffs)
    
    # Phase 1: Global search with differential evolution
    if time.time() - _start_time < _DEADLINE - _SAFETY_MARGIN - 60:
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        result_de = differential_evolution(
            _objective, bounds,
            seed=42, maxiter=100, popsize=15,
            tol=1e-8, polish=True
        )
        if result_de.success and result_de.fun < best_val:
            best_val = result_de.fun
            best_coeffs = result_de.x
    
    # Phase 2: Local refinement with L-BFGS-B
    if time.time() - _start_time < _DEADLINE - _SAFETY_MARGIN - 10:
        # Try multiple local starts
        starts = [
            best_coeffs.copy(),
            np.array([-1.0, 0.1, 0.01]),
            np.array([-1.2, 0.05, 0.001]),
            np.array([-0.5, 0.2, -0.01]),
            np.array([-2.0, 0.3, 0.05]),
        ]
        for start in starts:
            if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
                break
            try:
                res = minimize(
                    _objective, start,
                    method='L-BFGS-B',
                    bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish
    if time.time() - _start_time < _DEADLINE - _SAFETY_MARGIN - 5:
        try:
            res_nm = minimize(
                _objective, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_coeffs = res_nm.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
