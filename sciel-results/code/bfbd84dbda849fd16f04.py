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


def _compute_hessian(coeffs, Y, w_theta, w_phi):
    """
    Compute approximate Hessian of the L4/L2 ratio objective.
    Uses second variation of the quartic term.
    """
    f = np.tensordot(coeffs, Y, axes=([0], [2]))
    w_full = w_theta[:, np.newaxis] * w_phi
    
    f2 = f * f
    norm2_sq = np.sum(w_full * f2)
    norm4 = np.sum(w_full * f2 * f2)
    
    if norm2_sq < 1e-30:
        return np.eye(21)
    
    norm2 = np.sqrt(norm2_sq)
    norm4_val = norm4 ** 0.25
    
    # Hessian of ||f||_4^4 / ||f||_2^4 with respect to coefficients
    # Let g(c) = (sum w f^4)^(1/4) / (sum w f^2)^(1/2)
    # We approximate the Hessian using the structure of the problem
    
    # For the quartic part: d^2/dc_i dc_j (sum w f^4) = 12 * sum w f^2 Y_i Y_j
    # For the quadratic part: d^2/dc_i dc_j (sum w f^2) = 2 * sum w Y_i Y_j
    
    # Build the Hessian matrix
    H = np.zeros((21, 21))
    
    # Compute Y^T W Y and Y^T W f^2 Y
    Yf = Y  # shape (n_theta, n_phi, 21)
    
    # sum w f^2 Y_i Y_j
    f2_w = w_full * f2  # shape (n_theta, n_phi)
    Yf2 = Y * f2_w[:, :, np.newaxis]  # shape (n_theta, n_phi, 21)
    H_quad = np.einsum('ij,ijk,ijl->kl', w_full, Y, Y)
    
    # sum w f^4 Y_i Y_j / norm4_val^3
    f4_w = w_full * f2 * f2
    Yf4 = Y * f4_w[:, :, np.newaxis]
    H_quartic = np.einsum('ij,ijk,ijl->kl', w_full, Y, Y) * 0  # placeholder
    
    # More carefully:
    # H_ij = d^2 R / dc_i dc_j
    # R = A/B where A = (sum w f^4)^(1/4), B = (sum w f^2)^(1/2)
    # dA/dc_i = (1/4) (sum w f^4)^(-3/4) * 4 sum w f^3 Y_i = A^(-3) sum w f^3 Y_i
    # d^2A/dc_i dc_j = -3 A^(-5) (dA/dc_i)(dA/dc_j) + A^(-3) sum w 3 f^2 Y_i Y_j
    #                  = -3 A^(-5) sum_w_f3Y_i sum_w_f3Y_j + 3 A^(-3) sum w f^2 Y_i Y_j
    # dB/dc_i = (1/2) (sum w f^2)^(-1/2) * 2 sum w f Y_i = B^(-1) sum w f Y_i
    # d^2B/dc_i dc_j = -B^(-3) sum_w_fY_i sum_w_fY_j + B^(-1) sum w Y_i Y_j
    
    f3 = f2 * f
    sum_w_f3Y = np.einsum('ij,ijh->h', w_full * f3, Y)
    sum_w_fY = np.einsum('ij,ijh->h', w_full * f, Y)
    sum_w_f2Y = np.einsum('ij,ijh->h', w_full * f2, Y)
    
    A = norm4_val
    B = norm2
    
    # d^2A
    d2A = -3 * A**(-5) * np.outer(sum_w_f3Y, sum_w_f3Y) + 3 * A**(-3) * H_quad
    
    # d^2B
    H_quad_simple = np.einsum('ij,ijk,ijl->kl', w_full, Y, Y)
    d2B = -B**(-3) * np.outer(sum_w_fY, sum_w_fY) + B**(-1) * H_quad_simple
    
    # R = A/B
    # d^2R = d^2A/B - 2 dA dB/B^2 + A d^2B/B^2
    dA = A**(-3) * sum_w_f3Y  # this is actually dR/dA * ... let me redo
    
    # Actually, let's just use finite differences for the Hessian approximation
    # or use a simpler approach: L-BFGS-B with more iterations and tighter tolerance
    
    # For now, return identity as fallback
    return np.eye(21)


def _projected_newton_step(c, Y, w_theta, w_phi, max_iter=100, tol=1e-12):
    """
    Projected Newton method on the unit sphere.
    Minimize R(c) subject to ||c|| = 1.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    
    f_val, grad = _compute_ratio_and_grad(c, Y, w_theta, w_phi)
    
    # Project gradient onto tangent space
    grad_tan = grad - np.dot(grad, c) * c
    
    for it in range(max_iter):
        if np.linalg.norm(grad_tan) < tol:
            break
        
        # Newton direction: solve (H - lambda I) d = -grad_tan in tangent space
        # Approximate Hessian using L-BFGS-like approach
        # For simplicity, use gradient descent with adaptive step size
        
        d = -grad_tan
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        d = d / d_norm
        
        # Line search
        best_c = c
        best_f = f_val
        best_alpha = 0.1
        
        for alpha in [0.5, 0.25, 0.1, 0.05, 0.01]:
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
        
        if best_alpha == 0.1 and best_f >= f_val:
            # No improvement, break
            break
        
        c = best_c
        f_val = best_f
        
        f_val, grad = _compute_ratio_and_grad(c, Y, w_theta, w_phi)
        grad_tan = grad - np.dot(grad, c) * c
    
    return c, f_val


def _high_precision_polish(c, Y, w_theta, w_phi, max_iter=1000, tol=1e-15):
    """
    Very high-precision polish using L-BFGS-B with extremely tight tolerances.
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
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 0.01, 'maxls': 50}
        )
        if result.fun < obj(c):
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
    
    # Phase 1: Multiple starting points with L-BFGS-B
    np.random.seed(42)
    
    starts = [x0.copy()]
    
    # Random starts
    for _ in range(20):
        s = np.random.randn(21)
        s = s / np.linalg.norm(s)
        starts.append(s)
    
    # Structured: emphasize certain m values
    for m_pattern in range(8):
        s = np.zeros(21)
        s[10] = 1.0  # m=0
        m_val = 1 + (m_pattern % 5)
        s[10 + m_val] = 1.0  # m cos
        s[10 - m_val] = 1.0  # m sin
        if m_pattern >= 4:
            m_val2 = 2 + (m_pattern % 3)
            s[10 + m_val2] = 0.5  # another m
            s[10 - m_val2] = 0.5
        s = s / np.linalg.norm(s)
        starts.append(s)
    
    for s in starts:
        if time.time() > deadline - 120:
            break
        
        # L-BFGS-B with tight tolerance
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                s,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Phase 2: Projected Newton polish on best solution
    if time.time() < deadline - 90:
        best_coeffs, pn_ratio = _projected_newton_step(best_coeffs, Y, w_theta, w_phi, max_iter=200, tol=1e-13)
        if pn_ratio < best_ratio:
            best_ratio = pn_ratio
    
    # Phase 3: More random restarts
    for i in range(20):
        if time.time() > deadline - 60:
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
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Phase 4: High-precision final polish
    if time.time() < deadline - 30:
        best_coeffs = _high_precision_polish(best_coeffs, Y, w_theta, w_phi, max_iter=2000, tol=1e-16)
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
