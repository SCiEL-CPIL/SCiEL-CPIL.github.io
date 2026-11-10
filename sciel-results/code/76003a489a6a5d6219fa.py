# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize


def _build_basis():
    """Precompute quadrature points and the real orthonormal degree-10 basis matrix.
    Returns (M, weights, phi) where M[i, j] = Y_j at point i, j = m + 10.
    """
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes/weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # cos theta nodes
    cth = x
    # weights for dOmega = sin(theta) dtheta dphi; with u = cos theta, du = -sin theta dtheta
    # so integral over theta of g(cos theta) sin theta dtheta = integral_{-1}^{1} g(u) du
    # total: dOmega = dphi du, measure mass = 2pi * 2 = 4pi. Good.
    w_theta = w  # weights for the u-integral

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    # Full weights: w_theta[i] * dphi
    W = np.outer(w_theta, np.full(n_phi, dphi))

    # Build basis matrix M of shape (n_theta*n_phi, 21)
    N = n_theta * n_phi
    M = np.empty((N, 2 * k + 1), dtype=np.float64)

    # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # Condon-Shortley phase is in lpmv (scipy's lpmv includes (-1)^m).
    # P_10^m(cos theta) = lpmv(m, 10, theta) where theta is polar angle.
    # lpmv(m, l, theta) = P_l^m(cos theta) with Condon-Shortley phase.

    # Precompute P_10^m at each theta node for m = 0..10
    theta = np.arccos(np.clip(cth, -1.0, 1.0))
    P = np.zeros((n_theta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = lpmv(m, k, theta)

    # Normalization factors
    def norm(m):
        # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        logval = np.log(21.0 / (4.0 * np.pi)) + (
            np.math.lgamma(k - m + 1) - np.math.lgamma(k + m + 1)
        )
        return np.sqrt(np.exp(logval))

    idx = 0
    for i in range(n_theta):
        for j in range(n_phi):
            p = phi[j]
            for m in range(0, k + 1):
                Nm = norm(m)
                Pm = P[i, m]
                if m == 0:
                    Y = Nm * Pm
                    M[idx, m] = Y
                else:
                    Yc = np.sqrt(2.0) * Nm * Pm * np.cos(m * p)
                    Ys = np.sqrt(2.0) * Nm * Pm * np.sin(m * p)
                    M[idx, m] = Yc
                    M[idx, k + m] = Ys  # m > 0 -> index m; m<0 -> index k+|m|
            idx += 1

    return M, W


# Precompute once at module load
_M, _W = _build_basis()
_Wflat = _W.ravel()  # shape (N,)
_Npts = _M.shape[0]


def _objective_and_grad(coeffs):
    """Return R = ||f||4/||f||2 and its gradient w.r.t. coeffs."""
    f = _M @ coeffs  # shape (N,)
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(_Wflat * f2)  # ||f||2^2
    if norm2_sq <= 0.0:
        return 1e30, np.zeros_like(coeffs)
    norm4_sq = np.sum(_Wflat * f4)  # ||f||4^4
    # R = ||f||4 / ||f||2 = (norm4_sq)^{1/4} / (norm2_sq)^{1/2}
    R = (norm4_sq ** 0.25) / (norm2_sq ** 0.5)

    # Gradient of R. Let A = norm4_sq, B = norm2_sq.
    # R = A^{1/4} B^{-1/2}
    # dR/dA = (1/4) A^{-3/4} B^{-1/2}
    # dR/dB = -(1/2) A^{1/4} B^{-3/2}
    # dA/dc = 4 sum W f^3 (M)  -> gradA = 4 * (W*f^3) @ M
    # dB/dc = 2 sum W f (M)     -> gradB = 2 * (W*f) @ M
    w_f3 = _Wflat * f2 * f
    w_f = _Wflat * f
    gradA = 4.0 * (w_f3 @ _M)
    gradB = 2.0 * (w_f @ _M)
    dR_dA = 0.25 * (norm4_sq ** (-0.75)) * (norm2_sq ** (-0.5))
    dR_dB = -0.5 * (norm4_sq ** 0.25) * (norm2_sq ** (-1.5))
    grad = dR_dA * gradA + dR_dB * gradB
    return R, grad


def rudin_problem10() -> np.ndarray:
    """Return 21 real coefficients (m=-10..10) minimizing the L4/L2 ratio of
    the degree-10 spherical harmonic, using L-BFGS-B with a deadline guard."""
    t_start = time.time()
    deadline = t_start + 600.0  # safety margin below 630s

    k = 10
    n = 2 * k + 1

    # Initial guess: all ones (the seed)
    best_coeffs = np.ones(n, dtype=np.float64)
    best_val, _ = _objective_and_grad(best_coeffs)

    # Try a few starting points and let L-BFGS-B optimize.
    rng = np.random.default_rng(12345)

    starts = [np.ones(n, dtype=np.float64)]
    # A few random starts (normalized-ish) to escape local minima
    for _ in range(8):
        s = rng.standard_normal(n)
        starts.append(s)

    for s0 in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                _objective_and_grad,
                s0.copy(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
            )
            val, _ = _objective_and_grad(res.x)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final polish on the best with tighter tolerance if time remains
    if time.time() < deadline - 20.0:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs.copy(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-12, "maxls": 100},
            )
            val, _ = _objective_and_grad(res.x)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Sanity / validity checks
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if best_coeffs.shape != (21,):
        best_coeffs = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    mx = np.max(np.abs(best_coeffs))
    if mx < 1e-3:
        best_coeffs = best_coeffs / max(mx, 1e-12) * 1.0
        mx = np.max(np.abs(best_coeffs))
        if mx < 1e-3:
            best_coeffs = np.ones(21, dtype=np.float64)
    if mx > 1e5:
        best_coeffs = best_coeffs / mx * 1.0

    return best_coeffs


# EVOLVE-BLOCK-END
