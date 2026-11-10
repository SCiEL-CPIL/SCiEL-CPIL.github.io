# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv


def _build_harmonics_matrix():
    """Build matrix A where row i is the 21 real degree-10 harmonics at quadrature point i."""
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x = cos(theta), w are weights for integral over cos(theta) in [-1,1]
    # Surface measure: dOmega = sin(theta) dtheta dphi = -d(cos theta) dphi
    # So integral over sphere = integral_{-1}^{1} integral_0^{2pi} F dphi dx
    # Total mass = 4*pi: sum w_i * (2*pi/n_phi * n_phi) ... let's set up properly
    # Actually: integral_{S^2} F dOmega = integral_{-1}^{1} integral_0^{2pi} F(x, phi) dphi dx
    # With GL: approx sum_i w_i * (1/n_phi) * sum_j F(x_i, phi_j) * (2*pi) ... 
    # More precisely: integral_0^{2pi} dphi = 2*pi, and we use n_phi equal points with weight 2*pi/n_phi each
    # So quadrature: sum_i w_i * (2*pi/n_phi) * sum_j F(x_i, phi_j)
    # Total mass check: sum_i w_i = 2, times 2*pi = 4*pi. Correct.

    # Build grid
    cos_theta = x  # shape (n_theta,)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)  # shape (n_phi,)
    cos_theta_grid, phi_grid = np.meshgrid(cos_theta, phi, indexing='ij')
    # shape: (n_theta, n_phi)

    # Weights for each point
    # weight_i_j = w_i * (2*pi / n_phi)
    weights = np.outer(w, np.ones(n_phi) * (2.0 * np.pi / n_phi))  # shape (n_theta, n_phi)

    # Compute the 21 real spherical harmonics at each point
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)

    # Precompute N_m for m = 0..10
    N = np.zeros(k + 1)
    for m in range(k + 1):
        # (10-m)!/(10+m)!
        # Use log for stability
        from math import lgamma
        log_ratio = lgamma(11 - m) - lgamma(11 + m)
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * np.exp(log_ratio))

    # Compute associated Legendre polynomials P_10^m(cos_theta) for m = 0..10
    # scipy.special.lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley phase
    # lpmv(m, k, x) = P_k^m(x)

    # For each m, compute P_10^m at all cos_theta values
    P_vals = np.zeros((k + 1, n_theta, n_phi))
    for m in range(k + 1):
        P_vals[m] = lpmv(m, k, cos_theta_grid)  # shape (n_theta, n_phi)

    # Build the harmonics matrix
    # A has shape (n_theta * n_phi, 21)
    n_points = n_theta * n_phi
    A = np.zeros((n_points, 2 * k + 1))

    for m in range(k + 1):
        idx_pos = m + k  # index for +m
        idx_neg = k - m  # index for -m (if m > 0)

        # Y_+m
        if m == 0:
            Y_pos = N[0] * P_vals[0]
        else:
            Y_pos = np.sqrt(2.0) * N[m] * P_vals[m] * np.cos(m * phi_grid)
        A[:, idx_pos] = Y_pos.flatten()

        # Y_-m (only for m > 0)
        if m > 0:
            Y_neg = np.sqrt(2.0) * N[m] * P_vals[m] * np.sin(m * phi_grid)
            A[:, idx_neg] = Y_neg.flatten()

    # Flatten weights
    w_flat = weights.flatten()

    return A, w_flat


def _compute_ratio(A, w, c):
    """Compute R = ||f||_4 / ||f||_2 for f = A @ c."""
    f = A @ c  # shape (n_points,)
    n2 = np.dot(f, f)
    n4 = np.sum(w * f**4) / np.sum(w)  # Wait, need to be careful with normalization
    # Actually: ||f||_2^2 = sum w_i f_i^2 (this gives the integral)
    # ||f||_4^4 = sum w_i f_i^4
    # But the task says norms under surface measure with total mass 4*pi
    # So ||f||_2 = sqrt(integral |f|^2 dOmega) = sqrt(sum w_i f_i^2)
    # ||f||_4 = (integral |f|^4 dOmega)^(1/4) = (sum w_i |f_i|^4)^(1/4)
    # R = ||f||_4 / ||f||_2
    n2 = np.dot(w, f**2)
    n4 = np.dot(w, f**4)
    if n2 < 1e-30:
        return np.inf
    return (n4 ** 0.25) / np.sqrt(n2)


def _riemannian_step(A, w, c, lr):
    """One Riemannian gradient step on the unit sphere."""
    c = c / np.linalg.norm(c)  # ensure on sphere
    f = A @ c

    n2 = np.dot(w, f**2)
    n4 = np.dot(w, f**4)
    if n2 < 1e-30 or n4 < 1e-30:
        return c

    norm2 = np.sqrt(n2)
    norm4 = n4 ** 0.25
    R = norm4 / norm2

    # Gradient of R w.r.t. c
    # R = ||f||_4 / ||f||_2
    # dR = (1/||f||_2) * d||f||_4 - (||f||_4/||f||_2^2) * d||f||_2
    # d||f||_2 = (1/(2||f||_2)) * d(f^T f) = (1/||f||_2) * f^T A dc
    #   so grad_c ||f||_2 = (A^T (w * f)) / ||f||_2  ... wait
    #   ||f||_2^2 = sum w_i f_i^2 = f^T diag(w) f = c^T A^T diag(w) A c
    #   grad_c ||f||_2 = (A^T diag(w) A c) / ||f||_2 = (A^T (w * f)) / ||f||_2
    # d||f||_4 = (1/||f||_4^3) * d(sum w_i f_i^4) = (1/||f||_4^3) * 4 * sum w_i f_i^3 A_i dc
    #   grad_c ||f||_4 = (4 / ||f||_4^3) * A^T (w * f^3)
    #   But ||f||_4^4 = sum w_i f_i^4, so d||f||_4 = (1/(4||f||_4^3)) * 4 * sum w_i f_i^3 A_i dc
    #   = (1/||f||_4^3) * A^T (w * f^3) dc
    #   So grad_c ||f||_4 = (1/||f||_4^3) * A^T (w * f^3)

    grad_norm4 = (1.0 / norm4**3) * (A.T @ (w * f**3))
    grad_norm2 = (1.0 / norm2) * (A.T @ (w * f))

    grad_R = (1.0 / norm2) * grad_norm4 - (norm4 / norm2**2) * grad_norm2

    # Project to tangent space: P = I - c c^T
    grad_tangent = grad_R - np.dot(grad_R, c) * c

    # Adaptive learning rate based on gradient magnitude
    grad_mag = np.linalg.norm(grad_tangent)
    if grad_mag < 1e-15:
        return c

    # Use a step size that scales with the ratio
    step = lr * grad_tangent / grad_mag  # normalized direction

    # Retraction: move and normalize
    c_new = c - step
    c_new = c_new / np.linalg.norm(c_new)
    return c_new


def _optimize(A, w, c0, max_iter, time_limit):
    """Riemannian gradient descent with adaptive step size."""
    c = c0 / np.linalg.norm(c0)
    best_c = c.copy()
    best_R = _compute_ratio(A, w, c)

    lr = 0.1
    stall_count = 0

    for it in range(max_iter):
        if time.time() > time_limit:
            break

        c_new = _riemannian_step(A, w, c, lr)
        R_new = _compute_ratio(A, w, c_new)

        if R_new < best_R:
            best_c = c_new.copy()
            best_R = R_new
            c = c_new
            stall_count = 0
        else:
            # Reject step, reduce learning rate
            lr *= 0.7
            stall_count += 1
            if stall_count > 30:
                # Try a small random perturbation
                noise = np.random.randn(21) * 0.05
                c_new = c + noise
                c_new = c_new / np.linalg.norm(c_new)
                R_new = _compute_ratio(A, w, c_new)
                if R_new < best_R:
                    best_c = c_new.copy()
                    best_R = R_new
                    c = c_new
                    stall_count = 0
                    lr = 0.05
                else:
                    c = c  # stay
                    if stall_count > 60:
                        break

    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m (m = -10..10) for a real degree-10 spherical harmonic
    that minimizes R = ||f||_4 / ||f||_2 on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    # Build the harmonic evaluation matrix
    A, w = _build_harmonics_matrix()

    # Total weight should be 4*pi
    total_w = np.sum(w)
    w = w / total_w * 4.0 * np.pi  # normalize so total mass = 4*pi

    # Multi-start Riemannian optimization
    n_restarts = 50
    best_c = None
    best_R = np.inf

    # First try: all ones (baseline)
    c0 = np.ones(21)
    c_opt, R_opt = _optimize(A, w, c0, max_iter=500, time_limit=deadline)
    if R_opt < best_R:
        best_R = R_opt
        best_c = c_opt

    # Random restarts
    np.random.seed(42)
    for i in range(n_restarts):
        if time.time() > deadline:
            break

        # Random initial point on sphere
        c0 = np.random.randn(21)
        c0 = c0 / np.linalg.norm(c0)

        c_opt, R_opt = _optimize(A, w, c0, max_iter=300, time_limit=deadline)
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt

    # Final polish with more iterations on best
    if time.time() < deadline - 30:
        c_final, R_final = _optimize(A, w, best_c, max_iter=1000, time_limit=deadline)
        if R_final < best_R:
            best_R = R_final
            best_c = c_final

    # Ensure valid output
    best_c = np.asarray(best_c, dtype=np.float64)
    if best_c.shape != (21,):
        best_c = np.ones(21)
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(21)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 0.01
    if max_abs > 1e5:
        best_c = best_c / max_abs * 1e4

    return best_c


# EVOLVE-BLOCK-END
