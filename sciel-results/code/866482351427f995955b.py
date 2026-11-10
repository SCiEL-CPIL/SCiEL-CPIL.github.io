# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time
from math import factorial, pi


def _build_basis(k=10):
    """
    Build real orthonormal spherical harmonic basis values on a fixed quadrature grid.
    Returns:
        Y: array of shape (2k+1, n_theta, n_phi)
        weights: array of shape (n_theta, n_phi) for surface integration
    """
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is cos(theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Surface measure weights
    # integral f dOmega = integral_0^{2pi} integral_{-1}^{1} f(x,phi) dx dphi
    # Gauss-Legendre for dx, uniform grid for dphi
    phi_weights = np.full(n_phi, 2 * np.pi / n_phi)
    weights = np.outer(w, phi_weights)

    coeffs = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        num = factorial(k - abs_m)
        den = factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * pi) * num / den)
        
        # P_10^m(cos theta) with Condon-Shortley phase
        P = lpmv(abs_m, k, x)  # shape (n_theta,)
        
        if m == 0:
            coeffs[k, :, :] = N_m * P[:, np.newaxis]
        elif m > 0:
            coeffs[k + m, :, :] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            # m < 0
            coeffs[k + m, :, :] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return coeffs, weights


def _compute_R(c, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2
    """
    f = np.tensordot(c, Y, axes=([0], [0]))  # shape (n_theta, n_phi)
    
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    if norm2_sq <= 0:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def _compute_R_and_grad(c, Y, weights):
    """
    Compute R and its gradient with respect to c.
    R = ||f||_4 / ||f||_2 = (sum w f^4)^{1/4} / (sum w f^2)^{1/2}
    """
    f = np.tensordot(c, Y, axes=([0], [0]))
    
    f2 = f * f
    f3 = f * f2
    f4 = f2 * f2
    
    norm2_sq = np.sum(f2 * weights)
    norm4_sq = np.sum(f4 * weights)  # This is ||f||_4^4
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(c)
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    R = norm4 / norm2
    
    # Gradient derivation:
    # Let A = ||f||_4^4 = sum w f^4, B = ||f||_2^2 = sum w f^2
    # R = A^{1/4} B^{-1/2}
    # dR/dc = (1/4) A^{-3/4} (dA/dc) B^{-1/2} - A^{1/4} (1/2) B^{-3/2} (dB/dc)
    # dA/dc_m = 4 sum w f^3 Y_m
    # dB/dc_m = 2 sum w f Y_m
    
    dA_dc = 4.0 * np.tensordot(weights * f3, Y, axes=([0, 1], [1, 2]))
    dB_dc = 2.0 * np.tensordot(weights * f, Y, axes=([0, 1], [1, 2]))
    
    A = norm4_sq
    B = norm2_sq
    
    dR_dc = (1.0/4.0) * A**(-3.0/4.0) * dA_dc / np.sqrt(B) - \
            A**(1.0/4.0) * (1.0/2.0) * B**(-3.0/2.0) * dB_dc
    
    return R, dR_dc


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic that minimize ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin for 600s limit
    
    k = 10
    Y, weights = _build_basis(k)
    
    # Objective and gradient functions
    def objective(c):
        return _compute_R(c, Y, weights)
    
    def objective_grad(c):
        return _compute_R_and_grad(c, Y, weights)
    
    # Multi-start strategy
    rng = np.random.default_rng(42)
    starts = []
    
    # 1. All ones
    starts.append(np.ones(2 * k + 1, dtype=np.float64))
    
    # 2. Single modes
    for m in range(-k, k+1):
        c = np.zeros(2*k+1)
        c[k+m] = 1.0
        starts.append(c)
    
    # 3. Symmetric pairs (cos + sin)
    for m in range(1, k+1):
        c = np.zeros(2*k+1)
        c[k+m] = 1.0
        c[k-m] = 1.0
        starts.append(c)
    
    # 4. Alternating signs for even m
    c = np.zeros(2*k+1)
    for m in range(0, k+1, 2):
        c[k+m] = 1.0
        if m > 0:
            c[k-m] = 1.0
    starts.append(c)
    
    # 5. Random starts
    for _ in range(20):
        c = rng.normal(size=2*k+1)
        starts.append(c)
    
    # 6. Low-m bias (often better for L4/L2)
    for _ in range(10):
        c = rng.normal(size=2*k+1)
        # Bias towards low m
        for i in range(2*k+1):
            m = i - k
            c[i] *= np.exp(-0.1 * abs(m))
        starts.append(c)
    
    best_R = np.inf
    best_c = None
    
    # First pass: quick evaluation of all starts
    for c0 in starts:
        if time.time() > deadline:
            break
        try:
            R = objective(c0)
            if np.isfinite(R) and R < best_R:
                best_R = R
                best_c = c0.copy()
        except Exception:
            pass
    
    if best_c is None:
        return np.ones(2*k+1)
    
    # Second pass: optimize from top starts
    # Sort starts by initial R and take top 15
    start_scores = []
    for c0 in starts:
        try:
            R = objective(c0)
            if np.isfinite(R):
                start_scores.append((R, c0))
        except Exception:
            pass
    
    start_scores.sort(key=lambda x: x[0])
    top_starts = [c for _, c in start_scores[:15]]
    
    # Add best_c to top starts if not already there
    if not any(np.allclose(c, best_c, atol=1e-10) for c in top_starts):
        top_starts.append(best_c.copy())
    
    from scipy.optimize import minimize
    
    for c0 in top_starts:
        if time.time() > deadline - 30:
            break
            
        try:
            # Use L-BFGS-B with gradient
            res = minimize(objective_grad, c0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_R and np.isfinite(res.fun):
                best_R = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_grad, best_c, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_R and np.isfinite(res.fun):
                best_R = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Verify validity
    if len(best_c) != 21:
        return np.ones(2*k+1)
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(2*k+1)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1.0 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1.0 / max_abs)
        
    return best_c


# EVOLVE-BLOCK-END
