import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize
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
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        P = lpmv(abs_m, k, x_gl)
        
        if m == 0:
            vals = N_m * P
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)
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
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    return norm4 / norm2

def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R and its gradient."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    R = norm4 / norm2
    
    # Gradient: dR/dc = (1/(N2*N4^3)) Y^T (w f^3) - (N4/N2^2)*(1/N2) Y^T (w f)
    # = (1/(N2*N4^3)) Y^T (w f^3) - (R/N2^2) Y^T (w f)
    term1 = Y.T @ (weights * f**3)
    term2 = Y.T @ (weights * f)
    grad = (1.0 / (norm2 * norm4**3)) * term1 - (R / norm2**2) * term2
    return R, grad

def _power_iteration_step(coeffs, Y, weights, n_iter=100):
    """
    Perform power iteration on the nonlinear operator T(f) = |f|^2 f.
    This drives the system towards a local minimum of the L4/L2 ratio.
    """
    f = Y @ coeffs
    norm2 = np.sqrt(np.dot(weights, f*f))
    if norm2 < 1e-12:
        return coeffs
    
    f = f / norm2
    
    for _ in range(n_iter):
        g = f * f * f
        coeffs_new = Y.T @ (weights * g)
        coeffs = coeffs_new
        
    return coeffs

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(123)
    
    # Strategy: Exploit symmetry structure. The L4/L2 ratio is rotationally invariant.
    # We can fix the "orientation" of the harmonic by setting c_10 (m=10, cos term) > 0 and c_9 (m=9, sin term) = 0.
    # This reduces the effective dimensionality and focuses search on the shape, not orientation.
    # We also try different symmetry classes:
    # 1. Fully symmetric (only even m, no sine terms)
    # 2. Axially symmetric (m=0 only)
    # 3. General but with fixed gauge (c_10 > 0, c_9 = 0)
    
    n_starts = 100
    
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
            
        # Different types of starts
        if i % 5 == 0:
            # Axially symmetric start (m=0 only)
            c = np.zeros(21)
            c[10] = 1.0  # m=0
        elif i % 5 == 1:
            # Even m only, no sine terms (fully symmetric)
            c = np.zeros(21)
            # m = 0, 2, 4, 6, 8, 10 -> indices 10, 12, 14, 16, 18, 20
            c[10] = rng.normal()
            c[12] = rng.normal()
            c[14] = rng.normal()
            c[16] = rng.normal()
            c[18] = rng.normal()
            c[20] = rng.normal()
        elif i % 5 == 2:
            # Gauge fixed: c_10 > 0, c_9 = 0
            c = rng.normal(size=21)
            c[19] = 0.0  # m=9, sin term
            if c[20] < 0:  # m=10, cos term
                c[20] = -c[20]
        elif i % 5 == 3:
            # Random start
            c = rng.normal(size=21)
        else:
            # Start with strong m=10 component
            c = rng.normal(size=21)
            c[20] += 5.0  # boost m=10
            
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            c = np.ones(21)
        else:
            c = c / c_norm
            
        # Run power iteration
        c_iter = _power_iteration_step(c, Y, weights, n_iter=150)
        
        val = _compute_ratio(c_iter, Y, weights)
        
        if val < best_val:
            best_val = val
            best_coeffs = c_iter.copy()
            
    # Polish the best candidate with L-BFGS-B
    if best_coeffs is not None and time.time() < deadline - 20.0:
        def objective(c):
            val, _ = _compute_ratio_and_grad(c, Y, weights)
            return val
        
        def gradient(c):
            _, g = _compute_ratio_and_grad(c, Y, weights)
            return g
        
        try:
            res = minimize(objective, best_coeffs, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    # Fallback
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    # Final validation and scaling
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1.0
        else:
            best_coeffs[10] = 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1.0

    return best_coeffs
