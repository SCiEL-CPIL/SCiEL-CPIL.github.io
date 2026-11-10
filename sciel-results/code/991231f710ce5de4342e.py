import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import differential_evolution, minimize
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio(coeffs, Y, weights):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    # ||f||_4 = (sum w |f|^4)^(1/4)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    def objective(coeffs):
        # Normalize to avoid scaling issues in DE, though ratio is invariant
        # But DE works better with bounded variables.
        # We just compute the ratio.
        return _compute_ratio(coeffs, Y, weights)

    best_coeffs = None
    best_val = np.inf
    
    # Strategy: Differential Evolution
    # This is a population-based global optimizer, distinct from Basin Hopping.
    
    # Bounds for coefficients. Since scale invariant, [-1, 1] is fine for search,
    # but we can use [-10, 10] to allow larger scales if needed (though not needed).
    bounds = [(-10.0, 10.0) for _ in range(21)]
    
    # Run DE with a time limit
    # popsize=20 means 20*21 = 420 individuals.
    # maxiter=500
    # We will run it and check time periodically?
    # DE doesn't support time limit directly, so we run a fixed number of iterations
    # and hope it's fast enough.
    # Each objective call is ~ O(9100) ops. 420 individuals * 500 iterations = 210,000 calls.
    # 210,000 * 10 us = 2.1 seconds. Very fast.
    # So we can run many iterations.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=1000,
            popsize=25,
            tol=1e-8,
            polish=True  # Use L-BFGS-B to polish the best solution
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
            
    except Exception as e:
        print(f"Differential Evolution failed: {e}")
        
    # If DE didn't run or failed, try a few random restarts with L-BFGS-B
    if best_coeffs is None or not np.isfinite(best_val):
        rng = np.random.default_rng(42)
        for i in range(10):
            x0 = rng.normal(size=21)
            x0 /= np.linalg.norm(x0)
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
            if time.time() > deadline - 30.0:
                break

    # Final polish with L-BFGS-B to ensure we are at a local minimum
    if best_coeffs is not None and time.time() < deadline - 10.0:
        res = minimize(objective, best_coeffs, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < best_val:
            best_coeffs = res.x
            best_val = res.fun

    # Validate
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Normalize to valid range if possible
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1.0
        else:
            return np.ones(21)
            
    return best_coeffs
