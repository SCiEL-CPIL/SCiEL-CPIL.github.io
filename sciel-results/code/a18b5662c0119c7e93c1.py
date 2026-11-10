import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize, newton
import time


def _build_quadrature():
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi


def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
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


def _eval_R(c, Y, weights):
    f = Y @ c
    f2 = f * f
    norm2_sq = np.dot(weights, f2)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(np.dot(weights, f2 * f2)))
    return norm4 / norm2


def _eval_R_and_grad(c, Y, weights):
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    norm2_sq = np.dot(weights, f2)
    if norm2_sq <= 1e-15:
        return 1e10, np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(np.dot(weights, f2 * f2)))
    term1 = Y.T @ (weights * f3)
    term2 = Y.T @ (weights * f)
    grad = term1 / (norm2 * norm4**3) - norm4 * term2 / (norm2**3)
    return norm4 / norm2, grad


def _newton_step(c, Y, weights, max_iter=50):
    """Newton refinement on the critical point equation."""
    for _ in range(max_iter):
        f = Y @ c
        f2 = f * f
        f3 = f2 * f
        norm2_sq = np.dot(weights, f2)
        if norm2_sq <= 1e-15:
            break
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(np.dot(weights, f2 * f2)))

        # Critical point condition: f = lambda * |f|^2 * f / ||f||_4^2
        # Equivalently: g(c) = f - R^2 * f * |f|^2 / ||f||_4^2 = 0 (up to scaling)
        # Use the gradient-based Newton: solve J delta = -g
        # Simpler: use the fact that at critical point, grad(R) = 0
        val, grad = _eval_R_and_grad(c, Y, weights)
        if np.max(np.abs(grad)) < 1e-14:
            break

        # Approximate Hessian via finite differences or use Gauss-Newton
        # For the ratio functional, the Hessian of R is complex.
        # Instead, use a line search on the negative gradient (quasi-Newton)
        step = 0.1
        for ls in range(30):
            c_new = c - step * grad
            val_new = _eval_R(c_new, Y, weights)
            if val_new < val:
                c = c_new
                break
            step *= 0.5
        else:
            break

    return c


def _power_iteration(c, Y, weights, n_iter=200, tol=1e-14):
    """
    Nonlinear power iteration: c_{k+1} = T(c_k) / ||T(c_k)||
    where T(c) = Y^T (weights * f * |f|^2) with f = Y @ c
    This drives toward a critical point of ||f||_4/||f||_2.
    """
    c = c / np.linalg.norm(c)
    for it in range(n_iter):
        f = Y @ c
        f2 = f * f
        f3 = f2 * f
        c_new = Y.T @ (weights * f3)
        c_new = c_new / np.linalg.norm(c_new)
        delta = np.linalg.norm(c_new - c)
        c = c_new
        if delta < tol:
            break
    return c


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0

    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi

    best_coeffs = None
    best_val = np.inf

    def try_candidate(c0):
        nonlocal best_coeffs, best_val
        if time.time() > deadline - 15.0:
            return
        c0 = np.asarray(c0, dtype=np.float64)
        c0 = c0 / np.linalg.norm(c0)
        # Phase 1: power iteration
        c = _power_iteration(c0, Y, weights, n_iter=100)
        # Phase 2: Newton/line-search refinement
        c = _newton_step(c, Y, weights, max_iter=30)
        # Phase 3: local L-BFGS-B polish
        try:
            res = minimize(lambda cc: _eval_R(cc, Y, weights),
                           c,
                           jac=lambda cc: _eval_R_and_grad(cc, Y, weights)[1],
                           method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            val = _eval_R(c, Y, weights)
            if val < best_val:
                best_val = val
                best_coeffs = c

    # Structured starts
    starts = []
    for m_val in range(0, 11, 2):
        c = np.zeros(21)
        c[m_val + 10] = 1.0
        starts.append(c)
        if m_val > 0:
            c2 = np.zeros(21)
            c2[m_val + 10] = 1.0
            c2[-m_val + 10] = 1.0
            starts.append(c2)
            c3 = np.zeros(21)
            c3[m_val + 10] = 1.0
            c3[-m_val + 10] = -1.0
            starts.append(c3)

    # Mixed zonal + sectorial
    for ratio in [0.3, 0.5, 0.7, 1.0, 1.4, 2.0]:
        c = np.zeros(21)
        c[10] = 1.0
        c[20] = ratio
        starts.append(c)
        c[10], c[20] = ratio, 1.0
        starts.append(c)

    # Random starts
    rng = np.random.default_rng(123)
    for _ in range(8):
        starts.append(rng.normal(size=21))

    for c0 in starts:
        if time.time() > deadline - 20.0:
            break
        try_candidate(c0)

    # Additional random starts if time permits
    if best_coeffs is not None and time.time() < deadline - 40.0:
        for _ in range(15):
            if time.time() > deadline - 15.0:
                break
            try_candidate(rng.normal(size=21))

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs
        else:
            best_coeffs[10] = 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs

    return best_coeffs
