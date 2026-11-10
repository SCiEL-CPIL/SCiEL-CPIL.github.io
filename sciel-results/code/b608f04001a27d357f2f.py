# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time


def _build_quadrature():
    """Build the 70-point Gauss-Legendre x 130-point phi quadrature."""
    n_theta, n_phi = 70, 130
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1,1] = cos(theta), w are weights
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    # surface element: dOmega = sin(theta) dtheta dphi = -dx dphi
    # integral over sphere = sum_i sum_j w_i * (2pi/n_phi) * f(x_i, phi_j)
    # Actually: integral = (2pi/n_phi) * sum_j sum_i w_i * f(x_i, phi_j)
    # because sin(theta) dtheta = -dx, and theta from 0 to pi maps x from 1 to -1
    # so integral = (2pi/n_phi) * sum_j sum_i w_i * f(x_i, phi_j)
    weight_factor = 2 * np.pi / n_phi
    return x, w, phi, weight_factor


def _spherical_harmonics(x, phi, k=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m=-k..k.
    Y_0 = N_0 * P_10(cos theta)
    Y_m = sqrt(2) * N_|m| * P_k^{|m|}(cos theta) * cos(m*phi) for m > 0
    Y_m = sqrt(2) * N_|m| * P_k^{|m|}(cos theta) * sin(|m|*phi) for m < 0
    N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    """
    n = 2 * k + 1
    Y = np.zeros((n,) + x.shape + phi.shape, dtype=np.float64)
    
    for m_abs in range(k + 1):
        # Compute P_k^m(cos theta) with Condon-Shortley phase
        # scipy lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
        P = lpmv(m_abs, k, x)  # shape (n_theta,)
        
        # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # 21 = 2*(2*10+1)/4... wait, let me recompute
        # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        # = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        from math import factorial, lgamma
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + lgamma(k + 1 - m_abs) - lgamma(k + 1 + m_abs))
        N = np.exp(log_N)
        
        if m_abs == 0:
            Y[k] = N * P  # m=0 index is k=10
        else:
            # m > 0: index = k + m
            Y[k + m_abs] = np.sqrt(2.0) * N * P * np.cos(m_abs * phi)
            # m < 0: index = k - m
            Y[k - m_abs] = np.sqrt(2.0) * N * P * np.sin(m_abs * phi)
    
    return Y


def _evaluate_ratio(coeffs, x, w, phi, weight_factor):
    """Compute R = ||f||_4 / ||f||_2 using quadrature."""
    k = 10
    Y = _spherical_harmonics(x, phi, k)  # shape (21, n_theta, n_phi)
    
    # f = sum_m c_m Y_m
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # shape (n_theta, n_phi)
    
    # ||f||_2^2 = (2pi/n_phi) * sum_j sum_i w_i * f^2
    # ||f||_4^4 = (2pi/n_phi) * sum_j sum_i w_i * f^4
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = weight_factor * np.sum(w[:, None] * f2, axis=0).sum()
    norm4_4 = weight_factor * np.sum(w[:, None] * f4, axis=0).sum()
    
    if norm2_sq <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _evaluate_ratio_grad(coeffs, x, w, phi, weight_factor):
    """Compute R and its gradient w.r.t. coeffs."""
    k = 10
    Y = _spherical_harmonics(x, phi, k)  # shape (21, n_theta, n_phi)
    
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # shape (n_theta, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = weight_factor * np.sum(w[:, None] * f2, axis=0).sum()
    norm4_4 = weight_factor * np.sum(w[:, None] * f4, axis=0).sum()
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    R = norm4 / norm2
    
    # dR/dcoeffs
    # R = (integral f^4 / 4pi)^{1/4} / (integral f^2 / 4pi)^{1/2}
    # Let A = integral f^4, B = integral f^2 (these include the 4pi factor via weight_factor)
    # Actually let me use: ||f||_2^2 = B, ||f||_4^4 = A
    # R = A^{1/4} / B^{1/2}
    # dR/df = (1/4) A^{-3/4} * 4 f^3 / B^{1/2} - A^{1/4} * (1/2) B^{-3/2} * 2 f
    #       = A^{-3/4} f^3 / B^{1/2} - A^{1/4} f / B^{3/2}
    #       = (A^{-3/4} f^3 B - A^{1/4} f B^2) / (B^{3/2} * B)
    # Let me just compute numerically or use chain rule more carefully.
    
    # d(||f||_4)/dc = (1/4) ||f||_4^{-3} * integral 4 f^3 Y_m dc
    # d(||f||_2)/dc = (1/||f||_2) * integral f Y_m dc
    # dR/dc = (d||f||_4/dc * ||f||_2 - ||f||_4 * d||f||_2/dc) / ||f||_2^2
    
    # integral f^3 Y_m = sum over grid
    f3 = f2 * f
    int_f3_Y = np.zeros(21)
    int_f_Y = np.zeros(21)
    for m in range(21):
        int_f3_Y[m] = weight_factor * np.sum(w[:, None] * f3 * Y[m], axis=0).sum()
        int_f_Y[m] = weight_factor * np.sum(w[:, None] * f * Y[m], axis=0).sum()
    
    d_norm4_dc = 0.25 * norm4**(-3) * int_f3_Y
    d_norm2_dc = (1.0 / norm2) * int_f_Y
    
    grad = (d_norm4_dc * norm2 - norm4 * d_norm2_dc) / (norm2 * norm2)
    
    return R, grad


def _random_coefficients(rng):
    """Generate random valid coefficients."""
    coeffs = rng.standard_normal(21)
    # Ensure max|c| is in valid range
    max_c = np.max(np.abs(coeffs))
    if max_c < 1e-3:
        coeffs *= 1.0 / max_c
    elif max_c > 1e5:
        coeffs *= 1e5 / max_c
    return coeffs


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 600s safety margin
    
    x, w, phi, weight_factor = _build_quadrature()
    rng = np.random.default_rng(42)
    
    # Try multiple random starts with local optimization
    best_coeffs = np.ones(21, dtype=np.float64)
    best_ratio = _evaluate_ratio(best_coeffs, x, w, phi, weight_factor)
    
    n_starts = 5
    max_iter = 2000
    lr = 0.01
    lr_decay = 0.999
    
    for start_i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Initialize
        if start_i == 0:
            coeffs = np.ones(21, dtype=np.float64)
        else:
            coeffs = _random_coefficients(rng)
        
        curr_ratio = _evaluate_ratio(coeffs, x, w, phi, weight_factor)
        
        if curr_ratio < best_ratio:
            best_ratio = curr_ratio
            best_coeffs = coeffs.copy()
        
        # Gradient descent with adaptive step size
        current_lr = lr
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            ratio, grad = _evaluate_ratio_grad(coeffs, x, w, phi, weight_factor)
            
            if not np.isfinite(ratio):
                break
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = coeffs.copy()
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Normalize gradient for stable steps
            grad_normalized = grad / grad_norm
            
            # Try step
            new_coeffs = coeffs - current_lr * grad_normalized
            
            # Ensure validity
            max_c = np.max(np.abs(new_coeffs))
            if max_c < 1e-3 or max_c > 1e5 or not np.all(np.isfinite(new_coeffs)):
                current_lr *= 0.5
                continue
            
            new_ratio = _evaluate_ratio(new_coeffs, x, w, phi, weight_factor)
            
            if np.isfinite(new_ratio) and new_ratio < ratio:
                coeffs = new_coeffs
                current_lr = min(current_lr * 1.05, 0.1)
            else:
                current_lr *= 0.9
        
        # Local polish with smaller steps
        if time.time() > deadline - 30:
            break
        
        polish_lr = 0.001
        for it in range(500):
            if time.time() > deadline - 15:
                break
            
            ratio, grad = _evaluate_ratio_grad(coeffs, x, w, phi, weight_factor)
            if not np.isfinite(ratio):
                break
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = coeffs.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            grad_normalized = grad / grad_norm
            new_coeffs = coeffs - polish_lr * grad_normalized
            
            max_c = np.max(np.abs(new_coeffs))
            if max_c < 1e-3 or max_c > 1e5 or not np.all(np.isfinite(new_coeffs)):
                polish_lr *= 0.5
                continue
            
            new_ratio = _evaluate_ratio(new_coeffs, x, w, phi, weight_factor)
            if np.isfinite(new_ratio) and new_ratio < ratio:
                coeffs = new_coeffs
            else:
                polish_lr *= 0.8
    
    # Final validation
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs = np.ones(21, dtype=np.float64)
    elif max_c > 1e5:
        best_coeffs = best_coeffs * 1e5 / max_c
    
    return best_coeffs


# EVOLVE-BLOCK-END
