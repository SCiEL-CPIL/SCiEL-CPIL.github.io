"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes A(g) * A(F[g]) = [∫|P(x)e^{-x²/2}dx]² subject to P(0) = 0,
where P(x) = a0·H0 + a4·H4 + a8·H8 + a12·H12 and a12 is fixed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195.0  # 195s safety margin before 210s kill

# Precompute Hermite polynomial values on a grid
_N_GRID = 20001
_X_GRID = np.linspace(-15, 15, _N_GRID)
_DX = _X_GRID[1] - _X_GRID[0]

# Physicists' Hermite polynomials
_H0 = np.ones_like(_X_GRID)
_H4 = 16.0 * _X_GRID**4 - 48.0 * _X_GRID**2 + 12.0
_H8 = (256.0 * _X_GRID**8 - 3584.0 * _X_GRID**6 + 13440.0 * _X_GRID**4 -
       13440.0 * _X_GRID**2 + 1680.0)
_H12 = (4096.0 * _X_GRID**12 - 212992.0 * _X_GRID**10 + 3317760.0 * _X_GRID**8
        - 25333440.0 * _X_GRID**6 + 100810080.0 * _X_GRID**4 - 201620160.0 * _X_GRID**2
        + 798336.0)

# Gaussian weight
_GAUSS = np.exp(-_X_GRID**2 / 2.0)

# Values at zero
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 798336.0


def _compute_A_g(a0, a4, a8, smooth_eps=1e-8):
    """
    Compute A(g) = ∫|P(x)·e^{-x²/2}| dx using smooth absolute value.
    Returns (A_g, P_values) for potential reuse.
    """
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    P = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = P * _GAUSS
    
    # Smooth absolute value: |x| ≈ x*tanh(x/eps) for small eps
    # This is C∞ and approximates |x| well
    if smooth_eps > 0:
        abs_g = g * np.tanh(g / smooth_eps)
    else:
        abs_g = np.abs(g)
    
    A_g = np.trapezoid(abs_g, _X_GRID) if hasattr(np, 'trapezoid') else np.trapz(abs_g, _X_GRID)
    
    if not np.isfinite(A_g) or A_g < 0:
        return 1e10, None
    return A_g, P


def _l1_product(a0, a4, a8, smooth_eps=1e-8):
    """Compute [A(g)]² with smooth absolute value."""
    A_g, _ = _compute_A_g(a0, a4, a8, smooth_eps)
    return A_g * A_g


def _l1_product_exact(a0, a4, a8):
    """Compute [A(g)]² with exact absolute value."""
    A_g, _ = _compute_A_g(a0, a4, a8, smooth_eps=0.0)
    return A_g * A_g


def _adaptive_grid_evaluate(a0, a4, a8):
    """
    Evaluate A(g) with adaptive grid refinement near zeros of P(x).
    This gives a more accurate estimate for the final polish.
    """
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Coarse evaluation first
    P_coarse = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g_coarse = P_coarse * _GAUSS
    abs_g_coarse = np.abs(g_coarse)
    
    # Find sign changes to locate zeros
    signs = np.sign(P_coarse)
    zero_indices = np.where(np.diff(signs) != 0)[0]
    
    # Build adaptive grid
    x_fine = [_X_GRID[0]]
    for i in range(len(_X_GRID) - 1):
        if i in zero_indices:
            # Refine around zero crossing
            x_lo = _X_GRID[i]
            x_hi = _X_GRID[i + 1]
            local_grid = np.linspace(x_lo, x_hi, 100)
            x_fine.extend(local_grid)
        else:
            x_fine.append(_X_GRID[i + 1])
    
    x_fine = np.array(x_fine)
    
    # Evaluate on fine grid
    x2 = x_fine**2
    H0_f = np.ones_like(x_fine)
    H4_f = 16.0 * x2**2 - 48.0 * x2 + 12.0
    H8_f = (256.0 * x2**4 - 3584.0 * x2**3 + 13440.0 * x2**2 - 13440.0 * x2 + 1680.0)
    H12_f = (4096.0 * x2**6 - 212992.0 * x2**5 + 3317760.0 * x2**4
             - 25333440.0 * x2**3 + 100810080.0 * x2**2 - 201620160.0 * x2 + 798336.0)
    
    P_fine = a0 * H0_f + a4 * H4_f + a8 * H8_f + a12 * H12_f
    g_fine = P_fine * np.exp(-x2 / 2.0)
    abs_g_fine = np.abs(g_fine)
    
    A_g = np.trapezoid(abs_g_fine, x_fine) if hasattr(np, 'trapezoid') else np.trapz(abs_g_fine, x_fine)
    
    if not np.isfinite(A_g) or A_g < 0:
        return 1e10
    return A_g * A_g


def construct():
    """
    Optimize coefficients a0, a4, a8 to minimize A(g)*A(F[g]) = [A(g)]².
    Returns the three coefficients as a numpy array.
    """
    # Seed values from the initial program
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = seed_result.copy()
    best_bound = _l1_product_exact(*seed_result)
    
    def remaining():
        return _DEADLINE - time.time()
    
    # Phase 1: Global search with differential evolution on smooth surrogate
    if remaining() > 80:
        try:
            bounds = [(-3.0, 3.0), (-0.3, 0.3), (-0.005, 0.005)]
            de_result = differential_evolution(
                _l1_product,
                bounds,
                args=(1e-6,),
                seed=42,
                maxiter=200,
                tol=1e-10,
                polish=False,
                init='sobol',
                popsize=15,
                update_interval=1
            )
            if de_result.fun < best_bound:
                # Verify with exact evaluation
                exact_val = _l1_product_exact(*de_result.x)
                if exact_val < best_bound:
                    best_bound = exact_val
                    best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starts
    if remaining() > 40:
        starts = [
            best_result,
            seed_result,
            np.array([-1.0, 0.04, 0.0004]),
            np.array([-1.2, 0.05, 0.0005]),
            np.array([-0.8, 0.03, 0.0003]),
            np.array([-1.5, 0.06, 0.0006]),
            np.array([-0.5, 0.02, 0.0002]),
            np.array([-2.0, 0.08, 0.0008]),
        ]
        for start in starts:
            if remaining() < 30:
                break
            try:
                res = minimize(
                    _l1_product,
                    start,
                    args=(1e-7,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_bound:
                    exact_val = _l1_product_exact(*res.x)
                    if exact_val < best_bound:
                        best_bound = exact_val
                        best_result = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Gradient-based polish with L-BFGS-B on smooth surrogate
    if remaining() > 15:
        try:
            res = minimize(
                _l1_product,
                best_result,
                args=(1e-9,),
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_bound:
                exact_val = _l1_product_exact(*res.x)
                if exact_val < best_bound:
                    best_bound = exact_val
                    best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: High-precision polish with adaptive grid evaluation
    if remaining() > 5:
        try:
            # Use adaptive grid for more accurate evaluation
            res = minimize(
                _adaptive_grid_evaluate,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final verification with exact evaluation
    final_val = _l1_product_exact(*best_result)
    if not np.isfinite(final_val) or final_val >= best_bound:
        # Try seed as fallback
        seed_val = _l1_product_exact(*seed_result)
        if np.isfinite(seed_val) and seed_val < final_val:
            return seed_result
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
