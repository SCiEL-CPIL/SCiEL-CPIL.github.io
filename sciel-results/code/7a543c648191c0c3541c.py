# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time
import math


def _compute_ratio(coeffs):
    """Compute R = ||f||_4 / ||f||_2 for given coefficients."""
    k = 10
    # Gauss-Legendre quadrature points in cos(theta)
    x, w = np.polynomial.legendre.leggauss(70)
    # Equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Precompute factorials for normalization
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    f_vals = np.zeros((70, n_phi))
    
    for i, m in enumerate(range(-10, 11)):
        c = coeffs[i]
        if abs(c) < 1e-15:
            continue
        
        abs_m = abs(m)
        # Condon-Shortley phase is included in lpmv
        P = lpmv(abs_m, k, x)  # shape (70,)
        N_m = math.sqrt(21.0 / (4 * math.pi) * math.factorial(k - abs_m) / math.factorial(k + abs_m))
        
        if m == 0:
            Y_theta = N_m * P  # shape (70,)
            f_vals += c * Y_theta[:, np.newaxis]
        else:
            Y_theta = np.sqrt(2) * N_m * P  # shape (70,)
            if m > 0:
                Y_phi = np.cos(m * phi)  # shape (130,)
            else:
                Y_phi = np.sin(abs_m * phi)  # shape (130,)
            f_vals += c * Y_theta[:, np.newaxis] * Y_phi[np.newaxis, :]
    
    # Compute norms using quadrature
    weight_phi = 2 * np.pi / n_phi
    weights = w[:, np.newaxis] * weight_phi
    
    f2 = np.sum(f_vals**2 * weights)
    f4 = np.sum(f_vals**4 * weights)
    
    norm2 = np.sqrt(max(f2, 0))
    norm4 = np.sqrt(max(f4, 0))  # ||f||_4 = (integral |f|^4 dOmega)^(1/4)
    
    if norm2 < 1e-15:
        return np.inf
    
    return norm4 / norm2


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic that minimize
    R = ||f||_4 / ||f||_2.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Initial guess: all ones
    best_coeffs = np.ones(n_coeffs, dtype=np.float64)
    best_ratio = _compute_ratio(best_coeffs)
    
    # Use Nelder-Mead optimization with multiple restarts
    def objective(coeffs):
        if time.time() > deadline:
            return best_ratio
        return _compute_ratio(coeffs)
    
    rng = np.random.RandomState(42)
    
    # Start from the initial best
    x0 = best_coeffs.copy()
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Additional restarts if time permits
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        x0 = best_coeffs + rng.randn(n_coeffs) * 0.3
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True})
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Final validation
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs
    
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs, dtype=np.float64)
    
    return best_coeffs


# EVOLVE-BLOCK-END
