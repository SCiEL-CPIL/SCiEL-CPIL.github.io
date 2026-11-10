# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import math
import time

# Precompute quadrature and basis at module level for speed
_K = 10
_N_THETA = 70
_N_PHI = 130

def _setup():
    """Set up quadrature points and precompute basis functions."""
    # Gauss-Legendre nodes and weights in [-1, 1] for cos(theta)
    x, w = np.polynomial.legendre.leggauss(_N_THETA)
    
    # phi points
    phi = np.linspace(0, 2 * np.pi, _N_PHI, endpoint=False)
    
    # Build grid
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')  # shape (N_THETA, N_PHI)
    theta = np.arccos(cos_theta)
    
    # Surface element: dOmega = sin(theta) dtheta dphi
    # Since we're integrating over cos_theta, dOmega = -dcos_theta dphi
    # So weight for each point is w_i * (dphi) where dphi = 2*pi/N_PHI
    dphi = 2 * np.pi / _N_PHI
    # The integral over sphere: sum_i sum_j f(cos_theta_i, phi_j) * w_i * dphi
    # Total mass should be 4*pi
    # sum(w_i) = 2, so total = 2 * N_PHI * dphi = 2 * 2*pi = 4*pi. Good.
    
    # Precompute basis functions Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0(cos_theta)
    # Y_m = sqrt(2) * N_m * P_10^|m|(cos_theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_m * P_10^|m|(cos_theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    def _norm(m):
        m_abs = abs(m)
        # (10-m_abs)! / (10+m_abs)!
        log_ratio = math.lgamma(11 - m_abs) - math.lgamma(11 + m_abs)
        return math.sqrt(21.0 / (4.0 * math.pi) * math.exp(log_ratio))
    
    # Precompute P_10^m(cos_theta) for m = 0..10
    P = np.zeros((_K + 1, _N_THETA, _N_PHI))
    for m in range(_K + 1):
        # lpmv(m, 10, cos_theta) gives P_10^m(cos_theta) with Condon-Shortley phase
        P[m] = lpmv(m, 10, cos_theta)[:, np.newaxis] * np.ones((1, _N_PHI))
    
    # Build basis matrix: shape (21, N_THETA * N_PHI)
    n_pts = _N_THETA * _N_PHI
    basis = np.zeros((2 * _K + 1, n_pts))
    
    for m in range(-_K, _K + 1):
        idx = m + _K
        m_abs = abs(m)
        N_m = _norm(m)
        
        if m == 0:
            Y = N_m * P[0]
        elif m > 0:
            Y = math.sqrt(2.0) * N_m * P[m_abs] * np.cos(m * phi_grid)
        else:
            Y = math.sqrt(2.0) * N_m * P[m_abs] * np.sin(m_abs * phi_grid)
        
        basis[idx] = Y.flatten()
    
    # Quadrature weights vector: shape (n_pts,)
    # Each point (i, j) has weight w_i * dphi
    weights = np.kron(w, np.full(_N_PHI, dphi))
    
    return basis, weights, x, w, phi

# Precompute at module level
BASIS, WEIGHTS, X_GL, W_GL, PHI = _setup()
N_PTS = len(WEIGHTS)

def _compute_R(coeffs):
    """Compute R = ||f||_4 / ||f||_2 for given coefficients."""
    f = coeffs @ BASIS  # shape (N_PTS,)
    
    # ||f||_2^2 = sum f^2 * w
    f2 = f * f
    norm2_sq = np.sum(f2 * WEIGHTS)
    
    # ||f||_4^4 = sum f^4 * w
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * WEIGHTS)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # ||f||_4 = (sum f^4 w)^(1/4)
    
    return norm4 / norm2

def _compute_R_and_grad(coeffs):
    """Compute R and gradient of R with respect to coeffs.
    
    R = ||f||_4 / ||f||_2
    
    Let A = ||f||_4^4 = sum f^4 w, B = ||f||_2^2 = sum f^2 w
    R = A^(1/4) / B^(1/2)
    
    dR/dc_i = (1/4) A^(-3/4) * (4 f^3 * Y_i) * w / B^(1/2) - A^(1/4) * (1/2) B^(-3/2) * (2 f * Y_i) * w
            = A^(-3/4) * sum(f^3 * Y_i * w) / B^(1/2) - A^(1/4) * sum(f * Y_i * w) / B^(3/2)
    """
    f = coeffs @ BASIS
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    
    A = np.sum(f4 * WEIGHTS)  # ||f||_4^4
    B = np.sum(f2 * WEIGHTS)  # ||f||_2^2
    
    if B <= 0 or A <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    A_quarter = A ** 0.25
    A_inv_three_quarter = A ** (-0.75)
    B_half = np.sqrt(B)
    B_inv_three_half = B ** (-1.5)
    
    R = A_quarter / B_half
    
    # Gradient
    # dR/dc_i = A^(-3/4) * <f^3, Y_i>_w / B^(1/2) - A^(1/4) * <f, Y_i>_w / B^(3/2)
    grad_term1 = A_inv_three_quarter / B_half * (BASIS.T @ (f3 * WEIGHTS))
    grad_term2 = A_quarter * B_inv_three_half * (BASIS.T @ (f * WEIGHTS))
    grad = grad_term1 - grad_term2
    
    return R, grad

def _project_to_harmonic_space(f_grid):
    """Project a function on the sphere onto the degree-10 harmonic space.
    
    Returns coefficients c such that f = sum c_m Y_m minimizes ||f - sum c_m Y_m||_2^2.
    """
    # c_i = <f, Y_i>_w / <Y_i, Y_i>_w
    # Since Y_i are orthonormal, <Y_i, Y_i>_w = 1
    c = (BASIS.T @ (f_grid * WEIGHTS))
    return c

def rudin_problem10() -> np.ndarray:
    """Find coefficients minimizing L4/L2 ratio for degree-10 spherical harmonics."""
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    # Start with all-ones
    coeffs = np.ones(2 * _K + 1, dtype=np.float64)
    best_coeffs = coeffs.copy()
    best_R = _compute_R(best_coeffs)
    
    # Fixed-point iteration approach:
    # The Euler-Lagrange condition for minimizing ||f||_4/||f||_2 subject to f in H_10
    # is that f^3 is proportional to f (in the harmonic space), i.e., 
    # proj(f^3) = lambda * f
    # More precisely, we want to minimize R = ||f||_4/||f||_2
    # Taking derivative: the optimal f satisfies proj(f^3) = (3/4) * (||f||_4^4/||f||_2^2) * f / ||f||_2^2 ... 
    # Let me derive this more carefully.
    
    # Minimize J = ||f||_4^4 / ||f||_2^2 (equivalent to minimizing R^4)
    # J = A/B where A = sum f^4 w, B = sum f^2 w
    # dJ/dc_i = (4 f^3 Y_i w sum - A * 2 f Y_i w sum) / B^2
    # At optimum: 4 sum(f^3 Y_i w) * B - 2 A sum(f Y_i w) = 0
    # 2 B <f^3, Y_i> = A <f, Y_i> for all i
    # Since <f, Y_i> = c_i and <f^3, Y_i> = (proj f^3)_i
    # 2 B (proj f^3)_i = A c_i
    # So proj(f^3) = (A/(2B)) c = (A/(2B)) * (f projected back, but f IS in harmonic space)
    # So proj(f^3) = (A/(2B)) f
    
    # This gives us a fixed-point iteration:
    # Given f, compute g = f^3, project to get c_new, then f_new = c_new @ BASIS
    # But we need to scale properly. The condition is:
    # c_new = (A/(2B)) * c where c are the current coefficients
    # Actually: proj(f^3) = (A/(2B)) f means:
    # c_new = (A/(2B)) * c
    # But this just scales c, which doesn't change the ratio. We need to be more careful.
    
    # Actually, the fixed point should be:
    # c_new_i = <f^3, Y_i> / <f, Y_i> ... no that's not right either.
    
    # Let me think again. We want to minimize R(f) = ||f||_4/||f||_2 over f in H_10, f != 0.
    # Since R is homogeneous of degree 0, we can fix ||f||_2 = 1 and minimize ||f||_4.
    # Minimize A = sum f^4 w subject to sum f^2 w = 1.
    # Lagrangian: L = sum f^4 w - lambda (sum f^2 w - 1)
    # dL/dc_i = 4 sum(f^3 Y_i w) - 2 lambda sum(f Y_i w) = 0
    # 2 <f^3, Y_i> = lambda <f, Y_i> = lambda c_i
    # So <f^3, Y_i> = (lambda/2) c_i
    # This means proj(f^3) = (lambda/2) c = (lambda/2) f (since f = sum c_i Y_i)
    
    # Fixed point: f_new = proj(f^3) / ||proj(f^3)||_2 ... but we need to normalize.
    # Actually: if f is optimal, then f^3 = (lambda/2) f + h where h is orthogonal to H_10.
    # So f is proportional to proj(f^3).
    # Fixed point iteration: f_{n+1} = proj(f_n^3) / ||proj(f_n^3)||_2
    
    # But this might not converge well. Let's use a damped version.
    
    max_iters = 5000
    damping = 0.5
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        f = coeffs @ BASIS
        f3 = f * f * f
        
        # Project f^3 onto harmonic space
        c_new = (BASIS.T @ (f3 * WEIGHTS))
        
        # Normalize
        norm_new = np.sqrt(np.sum(c_new * c_new))  # ||f_new||_2 since orthonormal
        if norm_new < 1e-15:
            break
        c_new = c_new / norm_new
        
        # Damped update
        coeffs = (1 - damping) * coeffs + damping * c_new
        # Renormalize
        norm_c = np.sqrt(np.sum(coeffs * coeffs))
        if norm_c < 1e-15:
            break
        coeffs = coeffs / norm_c
        
        R = _compute_R(coeffs)
        if R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()
        
        # Check convergence
        if it > 10 and abs(R - best_R) < 1e-12:
            break
    
    # Now do gradient-based refinement
    coeffs = best_coeffs.copy()
    norm_c = np.sqrt(np.sum(coeffs * coeffs))
    if norm_c > 0:
        coeffs = coeffs / norm_c
    
    # Gradient descent with line search
    lr = 0.01
    for it in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        R, grad = _compute_R_and_grad(coeffs)
        
        if not np.isfinite(R):
            break
        
        # Normalize gradient
        grad_norm = np.sqrt(np.sum(grad * grad))
        if grad_norm < 1e-15:
            break
        
        # Gradient descent step
        coeffs_new = coeffs - lr * grad
        # Renormalize
        norm_new = np.sqrt(np.sum(coeffs_new * coeffs_new))
        if norm_new < 1e-15:
            break
        coeffs_new = coeffs_new / norm_new
        
        R_new = _compute_R(coeffs_new)
        
        if R_new < R:
            coeffs = coeffs_new
            if R_new < best_R:
                best_R = R_new
                best_coeffs = coeffs.copy()
        else:
            # Reduce learning rate
            lr *= 0.5
        
        if lr < 1e-10:
            break
    
    # Final high-precision polish
    coeffs = best_coeffs.copy()
    for it in range(100):
        if time.time() > deadline - 2.0:
            break
        
        R, grad = _compute_R_and_grad(coeffs)
        if not np.isfinite(R):
            break
        
        # Newton-like step using the fact that we're on a sphere
        # Project gradient to tangent space
        grad_tangent = grad - np.dot(grad, coeffs) * coeffs
        
        grad_norm = np.sqrt(np.sum(grad_tangent * grad_tangent))
        if grad_norm < 1e-14:
            break
        
        # Take a small step
        step = 1e-4 * grad_tangent / grad_norm
        coeffs_new = coeffs - step
        norm_new = np.sqrt(np.sum(coeffs_new * coeffs_new))
        if norm_new < 1e-15:
            break
        coeffs_new = coeffs_new / norm_new
        
        R_new = _compute_R(coeffs_new)
        if R_new < R:
            coeffs = coeffs_new
            if R_new < best_R:
                best_R = R_new
                best_coeffs = coeffs.copy()
    
    return best_coeffs

# EVOLVE-BLOCK-END
