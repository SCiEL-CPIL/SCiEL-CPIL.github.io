# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_theta = w
    w_phi = 2 * np.pi / n_phi
    return x, w_theta, phi, w_phi


def _spherical_harmonics(x, phi, degree=10):
    """
    Compute real orthonormal spherical harmonics of given degree.
    Returns array of shape (n_theta, n_phi, 2*degree+1) where index m corresponds to m-degree.
    """
    k = degree
    n_theta = len(x)
    n_phi = len(phi)
    
    norm_constants = np.zeros(k + 1)
    for m in range(k + 1):
        ratio = 1.0
        for j in range(10 - m + 1, 10 + m + 1):
            ratio /= j
        norm_constants[m] = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
    
    P = np.zeros((n_theta, k + 1))
    for m in range(k + 1):
        P[:, m] = lpmv(m, k, x)
    
    Y = np.zeros((n_theta, n_phi, 2 * k + 1))
    
    Y[:, :, k] = norm_constants[0] * P[:, 0][:, np.newaxis]
    
    for m in range(1, k + 1):
        cos_term = np.cos(m * phi)[np.newaxis, :]
        sin_term = np.sin(m * phi)[np.newaxis, :]
        P_m = P[:, m][:, np.newaxis]
        coeff = np.sqrt(2.0) * norm_constants[m]
        
        Y[:, :, k + m] = coeff * P_m * cos_term
        Y[:, :, k - m] = coeff * P_m * sin_term
    
    return Y


def _compute_ratio_and_grad(coeffs, Y, w_theta, w_phi):
    """
    Compute L4/L2 ratio and its gradient.
    coeffs: shape (21,)
    Y: shape (n_theta, n_phi, 21)
    w_theta: shape (n_theta,)
    w_phi: scalar
    
    Returns (ratio, grad)
    """
    f = np.tensordot(coeffs, Y, axes=([0], [2]))
    
    w_full = w_theta[:, np.newaxis] * w_phi
    
    f2 = f * f
    norm2_sq = np.sum(w_full * f2)
    
    f4 = f2 * f2
    norm4 = np.sum(w_full * f4)
    
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4_val = norm4 ** 0.25
    
    ratio = norm4_val / norm2
    
    f3 = f2 * f
    sum_w_f3Y = np.einsum('ij,ijh->h', w_full * f3, Y)
    sum_w_fY = np.einsum('ij,ijh->h', w_full * f, Y)
    
    A = norm4_val
    B = norm2
    
    grad = A**(-3) * B**(-1) * sum_w_f3Y - A * B**(-3) * sum_w_fY
    
    return ratio, grad


def _trust_region_step(c, Y, w_theta, w_phi, max_iter=200, tol=1e-12):
    """
    Trust-region method on the unit sphere for minimizing L4/L2 ratio.
    We work on the unit sphere: minimize R(c) subject to ||c|| = 1.
    Use a trust-region approach with line search.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    
    f_val, grad = _compute_ratio_and_grad(c, Y, w_theta, w_phi)
    
    # Project gradient onto tangent space
    grad_tan = grad - np.dot(grad, c) * c
    
    # Initial trust region radius
    delta = 1.0
    
    for it in range(max_iter):
        # Check convergence
        if np.linalg.norm(grad_tan) < tol:
            break
        
        # Compute search direction: negative projected gradient
        d = -grad_tan / np.linalg.norm(grad_tan)
        
        # Try different step sizes
        best_c = c
        best_f = f_val
        best_alpha = 0
        
        for alpha in [delta, delta * 0.5, delta * 0.25, delta * 0.1, delta * 0.05]:
            # Move along direction, then re-normalize to stay on sphere
            c_new = c + alpha * d
            norm_new = np.linalg.norm(c_new)
            if norm_new < 1e-10:
                continue
            c_new = c_new / norm_new
            
            f_new, _ = _compute_ratio_and_grad(c_new, Y, w_theta, w_phi)
            
            if f_new < best_f:
                best_f = f_new
                best_c = c_new
                best_alpha = alpha
        
        if best_alpha == 0:
            # Shrink trust region
            delta *= 0.5
            if delta < 1e-10:
                break
            continue
        
        # Update
        c = best_c
        f_val = best_f
        
        # Expand trust region if we took full step and improved
        if best_alpha == delta and f_val < f_val - 0.5 * np.dot(grad_tan, d) * best_alpha:
            delta = min(2 * delta, 2.0)
        elif best_alpha < delta:
            delta = best_alpha
        
        # Recompute gradient
        f_val, grad = _compute_ratio_and_grad(c, Y, w_theta, w_phi)
        grad_tan = grad - np.dot(grad, c) * c
    
    return c, f_val


def _rayleigh_polish(c, Y, w_theta, w_phi, max_iter=500, tol=1e-14):
    """
    High-precision polish using L-BFGS-B with very tight tolerances.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    
    def obj(c):
        return _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0]
    
    def jac(c):
        return _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1]
    
    try:
        result = minimize(
            obj,
            c,
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 0.1}
        )
        if result.fun < _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0]:
            c = result.x
    except Exception:
        pass
    
    return c / np.linalg.norm(c)


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Returns np.ndarray of shape (21,) with coefficients c_m for m = -10..10.
    """
    deadline = time.time() + 600
    
    # Build quadrature
    x, w_theta, phi, w_phi = _build_quadrature()
    Y = _spherical_harmonics(x, phi, degree=10)
    
    # Initial guess: all ones, normalized
    x0 = np.ones(21, dtype=np.float64)
    x0 = x0 / np.linalg.norm(x0)
    
    best_coeffs = x0.copy()
    best_ratio = _compute_ratio_and_grad(x0, Y, w_theta, w_phi)[0]
    
    # Phase 1: L-BFGS-B from multiple starting points
    np.random.seed(42)
    
    # Starting points: all-ones, random, and structured patterns
    starts = [x0.copy()]
    
    # Random starts
    for _ in range(10):
        s = np.random.randn(21)
        s = s / np.linalg.norm(s)
        starts.append(s)
    
    # Structured: emphasize certain m values
    for m_pattern in range(5):
        s = np.zeros(21)
        s[10] = 1.0  # m=0
        if m_pattern % 2 == 0:
            s[10 + 1] = 1.0  # m=1 cos
            s[10 - 1] = 1.0  # m=-1 sin
        else:
            s[10 + 2] = 1.0  # m=2 cos
            s[10 - 2] = 1.0  # m=-2 sin
        s = s / np.linalg.norm(s)
        starts.append(s)
    
    for s in starts:
        if time.time() > deadline - 90:
            break
        
        # L-BFGS-B
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                s,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Phase 2: Trust-region polish on best solution
    if time.time() < deadline - 60:
        best_coeffs, tr_ratio = _trust_region_step(best_coeffs, Y, w_theta, w_phi, max_iter=100, tol=1e-12)
        if tr_ratio < best_ratio:
            best_ratio = tr_ratio
    
    # Phase 3: More random restarts with trust-region
    for i in range(15):
        if time.time() > deadline - 45:
            break
        
        s = np.random.randn(21)
        s = s / np.linalg.norm(s)
        
        # Quick L-BFGS-B first
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                s,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
        
        # Trust-region polish
        if time.time() < deadline - 30:
            polished, pr_ratio = _trust_region_step(best_coeffs, Y, w_theta, w_phi, max_iter=50, tol=1e-12)
            if pr_ratio < best_ratio:
                best_ratio = pr_ratio
                best_coeffs = polished
    
    # Phase 4: High-precision final polish
    if time.time() < deadline - 20:
        best_coeffs = _rayleigh_polish(best_coeffs, Y, w_theta, w_phi, max_iter=500, tol=1e-14)
        best_ratio = _compute_ratio_and_grad(best_coeffs, Y, w_theta, w_phi)[0]
    
    # Final validation
    best_coeffs = best_coeffs / np.linalg.norm(best_coeffs)
    
    if len(best_coeffs) != 21:
        return x0
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return x0
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return x0
    
    return best_coeffs


# EVOLVE-BLOCK-END
