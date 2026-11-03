"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill

def _compute_c4_bound(roots, m=12):
    """
    Compute the C4 upper bound for the Fourier sign uncertainty.
    roots: 12 positive double-root locations.
    Uses the Laguerre-polynomial construction where P(x) = prod_i (x - r_i)^2.
    The C4 bound is the ratio of integrals that determines the uncertainty constant.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    # Build P(x)^2 = prod_i (x - r_i)^4
    # Use log-space for numerical stability
    x_max = max(roots) * 3 + 30
    x_min = 1e-8
    n_pts = 20000
    x = np.linspace(x_min, x_max, n_pts)
    
    log_P_sq = np.zeros_like(x)
    for r in roots:
        log_P_sq += 4.0 * np.log(np.abs(x - r) + 1e-30)
    
    log_P_sq = np.clip(log_P_sq, -700, 700)
    P_sq = np.exp(log_P_sq)
    
    # The C4 bound for Fourier sign uncertainty:
    # C4 = 2 * int_0^inf P(x)^2 dx / int_0^inf P(x)^2 * (1 - sin(x)/x) dx
    # This is a common form where the denominator involves the "sign kernel"
    
    # Use sinc: sin(x)/x = np.sinc(x/np.pi)
    kernel = 1.0 - np.sinc(x / np.pi)
    kernel = np.maximum(kernel, 1e-15)
    
    int_num = np.trapz(P_sq, x)
    int_den = np.trapz(P_sq * kernel, x)
    
    if int_den <= 0 or int_num <= 0:
        return 1e10
    
    c4 = 2.0 * int_num / int_den
    return c4

def _laguerre_roots(m):
    """Get the m positive roots of the Laguerre polynomial L_m(x)."""
    # L_m(x) coefficients
    coeffs = np.zeros(m + 1)
    for k in range(m + 1):
        coeffs[m - k] = ((-1) ** k) * (np.math.comb(m, k) / np.math.factorial(k))
    roots = np.roots(coeffs)
    roots = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots > 0]
    roots = np.sort(roots)
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    # Start with standard Laguerre roots
    roots0 = _laguerre_roots(12)
    
    if len(roots0) != 12:
        return [4.0 * (i + 1) for i in range(12)]
    
    best_roots = roots0.copy()
    best_c4 = _compute_c4_bound(best_roots)
    
    # Check time before optimizing
    elapsed = time.time() - _START_TIME
    if elapsed > 20:
        return [float(r) for r in best_roots]
    
    # Optimize root positions to minimize C4 bound
    try:
        # Use Nelder-Mead with adaptive simplex
        result = minimize(
            _compute_c4_bound,
            best_roots,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.x is not None:
            opt_roots = np.sort(result.x)
            if np.all(opt_roots > 0) and np.all(np.diff(opt_roots) > 0):
                opt_c4 = _compute_c4_bound(opt_roots)
                if opt_c4 < best_c4:
                    best_roots = opt_roots
                    best_c4 = opt_c4
    except Exception:
        pass
    
    # Check time before second round of optimization
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 30:
        try:
            # Second round with tighter tolerance
            result2 = minimize(
                _compute_c4_bound,
                best_roots,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result2.x is not None:
                opt_roots2 = np.sort(result2.x)
                if np.all(opt_roots2 > 0) and np.all(np.diff(opt_roots2) > 0):
                    opt_c42 = _compute_c4_bound(opt_roots2)
                    if opt_c42 < best_c4:
                        best_roots = opt_roots2
                        best_c4 = opt_c42
        except Exception:
            pass
    
    # Final safety check
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Return whatever we have
        pass
    
    best_roots = np.sort(np.maximum(best_roots, 1e-6))
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
