# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.integrate import fixed_quad


def _build_quadrature():
    """Build 70-point Gauss-Legendre (cos theta) x 130 phi quadrature."""
    N_theta = 70
    N_phi = 130
    # Gauss-Legendre nodes/weights on [-1, 1] for cos(theta)
    x, w = np.polynomial.legendre.leggauss(N_theta)
    # phi grid: equally spaced
    phi = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    dphi = 2.0 * np.pi / N_phi
    # theta from cos(theta) = x
    theta = np.arccos(x)
    # Surface measure: dOmega = sin(theta) dtheta dphi.
    # Change of variable u = cos(theta): dtheta = -du / sin(theta), so
    # sin(theta) dtheta dphi = du dphi. So integrate over u in [-1,1] and phi.
    # Total weight per (u, phi) cell: w_i * dphi
    return x, w, phi, dphi


def _spherical_harmonics_matrix(x, phi):
    """
    Build the real orthonormal degree-10 spherical harmonic values at
    points (x = cos theta, phi). Returns matrix Y of shape (N_theta, N_phi, 21)
    where column index i corresponds to m = i - 10.

    Definitions:
      Y_0 = N_0 P_10^0
      Y_m = sqrt(2) N_m P_10^m cos(m phi)   for m > 0
      Y_m = sqrt(2) N_|m| P_10^|m| sin(|m| phi) for m < 0
    where P_10^m = P_10^m(cos theta) with Condon-Shortley phase,
    N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!).
    """
    k = 10
    N_theta = len(x)
    N_phi = len(phi)
    Y = np.zeros((N_theta, N_phi, 2 * k + 1), dtype=np.float64)

    # Precompute P_10^m(x) for m = 0..10
    P = {}
    for m in range(0, k + 1):
        P[m] = lpmv(m, k, x)  # includes Condon-Shortley phase

    # Normalization constants
    N_norm = {}
    for m in range(0, k + 1):
        N_norm[m] = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                            np.math.factorial(k - m) / np.math.factorial(k + m))

    # m = 0
    Y[:, :, k] = N_norm[0] * P[0]  # broadcast over phi

    for m in range(1, k + 1):
        # m > 0: cos(m phi)
        Y[:, :, k + m] = np.sqrt(2.0) * N_norm[m] * P[m][:, None] * np.cos(m * phi)[None, :]
        # m < 0: sin(|m| phi)
        Y[:, :, k - m] = np.sqrt(2.0) * N_norm[m] * P[m][:, None] * np.sin(m * phi)[None, :]

    return Y


def _compute_f(Y, coeffs, x, w, phi, dphi):
    """Compute f values on the grid and the integral weights."""
    # coeffs shape (21,), Y shape (N_theta, N_phi, 21)
    f = np.einsum('ijm,m->ij', Y, coeffs)
    return f


def _objective_and_grad(coeffs, Y, x, w, phi, dphi):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient w.r.t. coeffs.
    Uses quadrature weights: weight at (i,j) = w_i * dphi.
    """
    f = np.einsum('ijm,m->ij', Y, coeffs)
    # Weights
    W = w[:, None] * dphi  # shape (N_theta, 1) broadcast to (N_theta, N_phi)

    f2 = f * f
    f4 = f2 * f2

    int_f2 = np.sum(f2 * W)
    int_f4 = np.sum(f4 * W)

    norm2 = np.sqrt(int_f2)
    norm4 = int_f4 ** 0.25

    if norm2 < 1e-15:
        return 1e15, np.zeros_like(coeffs)

    R = norm4 / norm2

    # Gradient of R = ||f||_4 / ||f||_2
    # d/dcoeffs [ (∫f^4)^(1/4) / (∫f^2)^(1/2) ]
    # Let A = ∫f^4, B = ∫f^2. R = A^(1/4) / B^(1/2)
    # dR/dcoeffs = (1/4) A^(-3/4) * dA/dcoeffs / B^(1/2) - A^(1/4) * (1/2) B^(-3/2) * dB/dcoeffs
    # dA/dcoeffs = ∫ 4 f^3 * (df/dcoeffs) dΩ = 4 ∫ f^3 Y dΩ  (vector over m)
    # dB/dcoeffs = ∫ 2 f * (df/dcoeffs) dΩ = 2 ∫ f Y dΩ
    #
    # df/dcoeffs_m = Y_m, so:
    # dA/dcoeffs_m = 4 ∫ f^3 Y_m dΩ
    # dB/dcoeffs_m = 2 ∫ f Y_m dΩ

    # Compute integrals via quadrature
    # int_f3_Y_m = ∫ f^3 * Y_m dΩ = sum_ij f_ij^3 * Y_ijm * W_ij
    f3 = f2 * f
    # Shape: f3 (N_theta, N_phi), Y (N_theta, N_phi, 21)
    # We need sum over (i,j) of f3_ij * Y_ijm * W_ij for each m
    # = einsum('ij,ijm,ij->m', f3, Y, W)
    int_f3_Y = np.einsum('ij,ijm,ij->m', f3, Y, W)
    int_f_Y = np.einsum('ij,ijm,ij->m', f, Y, W)

    dA = 4.0 * int_f3_Y
    dB = 2.0 * int_f_Y

    A = int_f4
    B = int_f2

    # Guard against zero
    if A < 1e-30:
        return R, np.zeros_like(coeffs)

    dR_dA = 0.25 * A ** (-0.75)
    dR_dB = -0.5 * A ** 0.25 * B ** (-1.5)

    grad = dR_dA * dA + dR_dB * dB

    return R, grad


def _normalize(coeffs):
    """Normalize so that ||f||_2 = 1 (approximately, via the quadratic form)."""
    # We can just scale by the L2 norm computed from the objective
    return coeffs


def _gradient_descent(coeffs0, Y, x, w, phi, dphi, max_iters=2000,
                      lr=0.01, tol=1e-10, time_limit=None, start_time=None):
    """
    Gradient descent on R(coeffs) with L2 normalization of f enforced by
    projecting back to the unit L2 sphere after each step.
    """
    coeffs = coeffs0.copy().astype(np.float64)
    best_R = _objective_and_grad(coeffs, Y, x, w, phi, dphi)[0]
    best_coeffs = coeffs.copy()

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break

        R, grad = _objective_and_grad(coeffs, Y, x, w, phi, dphi)

        if not np.isfinite(R):
            break

        if R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()

        # Normalization: we want to minimize R subject to ||f||_2 = 1.
        # Since R is scale-invariant, we can just do gradient descent on R
        # directly. But to keep coefficients in a reasonable range, we
        # normalize f to have ||f||_2 = 1 after each step.
        #
        # Actually, since R is scale-invariant, the gradient descent on R
        # will naturally stay on a scale-invariant manifold. Let's just
        # do plain gradient descent and normalize at the end.
        #
        # Better: use the fact that R is scale-invariant. We can fix
        # ||f||_2 = 1 and optimize on the unit sphere of coefficient space
        # (with the L2 norm defined by the spherical harmonic orthonormality).
        #
        # Since the Y_m are orthonormal, ||f||_2^2 = sum_m c_m^2.
        # So the constraint is sum c_m^2 = 1.
        #
        # We can do gradient descent on R projected onto the unit sphere.

        # Project gradient onto tangent space of unit sphere
        # (constraint: ||c|| = 1)
        c_norm = np.linalg.norm(coeffs)
        if c_norm < 1e-15:
            break
        # Tangent projection: grad - (grad . c / ||c||^2) * c
        grad_dot_c = np.dot(grad, coeffs)
        grad_tangent = grad - (grad_dot_c / (c_norm * c_norm)) * coeffs

        # Adaptive step size
        step = lr / (1.0 + it * 0.01)

        # Try the step
        new_coeffs = coeffs - step * grad_tangent

        # Renormalize to unit L2 norm
        new_norm = np.linalg.norm(new_coeffs)
        if new_norm < 1e-15:
            break
        new_coeffs = new_coeffs / new_norm

        new_R, _ = _objective_and_grad(new_coeffs, Y, x, w, phi, dphi)

        # Line search if needed
        if new_R > R:
            # Halve step
            for ls in range(10):
                step *= 0.5
                new_coeffs = coeffs - step * grad_tangent
                new_norm = np.linalg.norm(new_coeffs)
                if new_norm < 1e-15:
                    break
                new_coeffs = new_coeffs / new_norm
                new_R, _ = _objective_and_grad(new_coeffs, Y, x, w, phi, dphi)
                if new_R <= R:
                    break
            if new_R > R:
                # Accept anyway with small step to avoid stalling
                new_coeffs = coeffs - 0.001 * grad_tangent
                new_norm = np.linalg.norm(new_coeffs)
                if new_norm > 1e-15:
                    new_coeffs = new_coeffs / new_norm
                else:
                    break

        coeffs = new_coeffs

        if it > 10 and abs(R - best_R) < tol:
            break

    return best_coeffs, best_R


def _random_start(rng):
    """Generate a random unit-norm coefficient vector."""
    c = rng.standard_normal(21)
    c /= np.linalg.norm(c)
    return c


def _refine(coeffs, Y, x, w, phi, dphi, max_iters=3000, lr=0.005,
            tol=1e-12, time_limit=None, start_time=None):
    """High-precision polish pass with smaller learning rate."""
    return _gradient_descent(coeffs, Y, x, w, phi, dphi,
                             max_iters=max_iters, lr=lr, tol=tol,
                             time_limit=time_limit, start_time=start_time)


def rudin_problem10() -> np.ndarray:
    k = 10
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s limit

    # Build quadrature and spherical harmonic values
    x, w, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x, phi)

    # Pre-compute the orthonormality check: sum_ij Y_ijm * Y_ijm' * W_ij should be delta_mm'
    # (This is a sanity check, not needed for the algorithm.)

    rng = np.random.default_rng(42)

    best_coeffs = None
    best_R = np.inf

    # Multiple restarts
    n_restarts = 20
    max_iters_per_restart = 1500

    for r in range(n_restarts):
        if time.time() > deadline - 60.0:
            break

        c0 = _random_start(rng)

        # First, normalize c0 so that ||f||_2 = 1
        # Since Y_m are orthonormal, ||f||_2^2 = sum c_m^2, so c0 is already unit norm.

        coeffs, R = _gradient_descent(
            c0, Y, x, w, phi, dphi,
            max_iters=max_iters_per_restart, lr=0.02, tol=1e-10,
            time_limit=deadline - time.time() - 30.0,
            start_time=start_time
        )

        if np.isfinite(R) and R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()

    # Polish the best result
    if best_coeffs is not None and time.time() < deadline - 30.0:
        polished, R_polished = _refine(
            best_coeffs, Y, x, w, phi, dphi,
            max_iters=5000, lr=0.001, tol=1e-13,
            time_limit=deadline - time.time() - 10.0,
            start_time=start_time
        )
        if np.isfinite(R_polished) and R_polished < best_R:
            best_R = R_polished
            best_coeffs = polished

    if best_coeffs is None:
        # Fallback: all ones
        best_coeffs = np.ones(2 * k + 1, dtype=np.float64)

    # Ensure validity: max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)

    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * k + 1, dtype=np.float64)

    return best_coeffs


# EVOLVE-BLOCK-END
