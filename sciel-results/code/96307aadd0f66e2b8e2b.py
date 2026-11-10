# EVOLVE-BLOCK-START
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

def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 and its gradient."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    S2 = np.dot(weights, f2)
    S4 = np.dot(weights, f4)
    
    if S2 <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    # R = sqrt(S4/S2)
    # We minimize R^2 = S4/S2 to avoid sqrt in gradient
    R2 = S4 / S2
    R = np.sqrt(R2)
    
    # Gradient of R2 = S4/S2
    # d(S4/S2)/dc = (1/S2) dS4/dc - (S4/S2^2) dS2/dc
    # dS4/dc = 4 * Y^T (w * f^3)
    # dS2/dc = 2 * Y^T (w * f)
    
    grad_S4 = 4.0 * (Y.T @ (weights * f**3))
    grad_S2 = 2.0 * (Y.T @ (weights * f))
    
    grad_R2 = (1.0 / S2) * grad_S4 - (S4 / (S2 * S2)) * grad_S2
    
    # Gradient of R = 0.5 * R^-1 * dR2/dc
    # But we can just minimize R2 directly, or return grad of R.
    # Let's return grad of R for standard minimizers.
    grad_R = 0.5 / R * grad_R2
    
    return R, grad_R

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    
    n_gl = len(w_gl)
    n_phi = Y.shape[0] // n_gl
    weights = np.repeat(w_gl, n_phi) * dphi
    
    def objective_and_grad(c):
        R, g = _compute_ratio_and_grad(c, Y, weights)
        return R, g
    
    # Initial guess: all ones
    x0 = np.ones(2 * k + 1, dtype=np.float64)
    
    best_coeffs = x0.copy()
    best_val = objective_and_grad(x0)[0]
    
    n_restarts = 10
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Random initialization with some structure
            # Try to bias towards lower m or specific symmetries
            x_init = rng.normal(size=2 * k + 1)
            # Scale to avoid numerical issues
            x_init /= np.linalg.norm(x_init)
            x_init *= 1.0
        
        # Use L-BFGS-B with gradient
        try:
            res = minimize(
                lambda c: objective_and_grad(c)[0],
                x_init,
                jac=lambda c: objective_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                # Check validity
                c_opt = res.x
                if np.all(np.isfinite(c_opt)):
                    max_abs = np.max(np.abs(c_opt))
                    if max_abs >= 1e-3 and max_abs <= 1e5:
                        val = res.fun
                        if val < best_val:
                            best_val = val
                            best_coeffs = c_opt.copy()
        except Exception:
            continue
        
        # Optional: Refine with BFGS if time permits
        if time.time() < deadline - 10.0:
            try:
                res2 = minimize(
                    lambda c: objective_and_grad(c)[0],
                    best_coeffs,
                    jac=lambda c: objective_and_grad(c)[1],
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-10}
                )
                if res2.fun < best_val:
                    c_opt2 = res2.x
                    if np.all(np.isfinite(c_opt2)):
                        max_abs = np.max(np.abs(c_opt2))
                        if max_abs >= 1e-3 and max_abs <= 1e5:
                            best_val = res2.fun
                            best_coeffs = c_opt2.copy()
            except Exception:
                pass

    # Final validation
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs > 0:
            best_coeffs = best_coeffs * (1.0 / max_abs)
            best_coeffs = best_coeffs * 1.0 # Normalize to 1
        else:
            return np.ones(21)
    
    return best_coeffs

# EVOLVE-BLOCK-END
