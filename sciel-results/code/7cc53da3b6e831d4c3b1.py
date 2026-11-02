"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) computed via recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=np.float64)
    elif n == 1:
        return 2.0 * x
    else:
        h0 = np.ones_like(x, dtype=np.float64)
        h1 = 2.0 * x
        for k in range(2, n + 1):
            h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
            h0, h1 = h1, h2
        return h1

def _compute_bound(coeffs, x=None):
    """
    Compute A(f)*A(F) for g(x) = P(x)*exp(-x^2/2).
    P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12, c12 fixed by P(0)=0.
    Since all degrees are multiples of 4, F[g] = g, so bound = A_f^2.
    A_f = ∫x^2 g^2 dx / ∫g^2 dx
    """
    if x is None:
        x = np.linspace(-15, 15, 15001)
    dx = x[1] - x[0]
    
    c0, c4, c8 = coeffs
    # P(0) = c0*1 + c4*(-12) + c8*1680 + c12*(-103950) = 0
    c12 = -(c0 * 1.0 + c4 * (-12.0) + c8 * 1680.0) / (-103950.0)
    c12 = (c0 - 12.0*c4 + 1680.0*c8) / 103950.0
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g2 = P**2 * np.exp(-x**2)
    
    norm = np.trapz(g2, x)
    if norm <= 0 or not np.isfinite(norm):
        return np.inf
    
    moment2 = np.trapz(x**2 * g2, x)
    A_f = moment2 / norm
    
    return A_f * A_f  # = A_f^2 since A_F = A_f

def construct():
    deadline = time.time() + 190
    
    seed_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_coeffs = seed_coeffs.copy()
    best_bound = _compute_bound(seed_coeffs)
    
    # Global optimization with differential evolution
    try:
        result = differential_evolution(
            lambda c: _compute_bound(c),
            bounds=[(-3, 3), (-1, 1), (-0.01, 0.01)],
            seed=42,
            maxiter=30,
            tol=1e-10,
            polish=True,
            popsize=20
        )
        if np.isfinite(result.fun) and result.fun < best_bound:
            best_bound = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Local refinement from multiple starts
    if time.time() < deadline - 20:
        starts = [
            seed_coeffs,
            np.array([-1.0, 0.04, 0.0003]),
            np.array([-1.2, 0.05, 0.0004]),
            np.array([-0.8, 0.03, 0.0002]),
            np.array([-1.5, 0.06, 0.0005]),
            np.array([-1.1, 0.045, 0.00035]),
            np.array([-0.9, 0.035, 0.00025]),
            best_coeffs,
        ]
        for start in starts:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(
                    lambda c: _compute_bound(c),
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-13}
                )
                if np.isfinite(res.fun) and res.fun < best_bound:
                    best_bound = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Focused grid search
    if time.time() < deadline - 25:
        c0_range = np.linspace(-1.5, -0.5, 15)
        c4_range = np.linspace(0.02, 0.08, 12)
        c8_range = np.linspace(0.0001, 0.0008, 8)
        
        for c0 in c0_range:
            if time.time() > deadline - 20:
                break
            for c4 in c4_range:
                if time.time() > deadline - 15:
                    break
                for c8 in c8_range:
                    b = _compute_bound(np.array([c0, c4, c8]))
                    if np.isfinite(b) and b < best_bound:
                        best_bound = b
                        best_coeffs = np.array([c0, c4, c8])
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda c: _compute_bound(c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
