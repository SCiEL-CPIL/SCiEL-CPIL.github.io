# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv


def _make_grid():
    """Evaluator grid: 70-pt Gauss-Legendre in cos(theta) x 130 uniform phi.
    Returns (theta, phi, w) with surface-measure weights summing to 2*pi."""
    n_theta = 70
    n_phi = 130
    x, wgl = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta2, phi2 = np.meshgrid(theta, phi, indexing="ij")
    w = 0.5 * wgl[:, None] * np.ones((1, n_phi))
    return theta2, phi2, w


def _basis(k, theta, phi):
    """Real orthonormal degree-k spherical harmonics on the grid.
    Returns array of shape (2k+1, n) with c[i] -> m = i - k."""
    n = theta.size
    Y = np.empty((2 * k + 1, n), dtype=np.float64)
    for m in range(-k, k + 1):
        am = abs(m)
        P = lpmv(am, k, np.cos(theta))
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                    float(np.math.factorial(k - am)) /
                    float(np.math.factorial(k + am)))
        if m == 0:
            Y[k] = N * P
        elif m > 0:
            Y[k + m] = np.sqrt(2.0) * N * P * np.cos(m * phi)
        else:
            Y[k + m] = np.sqrt(2.0) * N * P * np.sin(am * phi)
    return Y


def _orrows(w, Y):
    """Rotate rows of Y to be L2-orthonormal under weights w (deterministic)."""
    M = w[None, :] @ Y
    M = M + 1e-12 * np.eye(M.shape[0])
    Q, R = np.linalg.qr(M.T)
    Q = Q * np.sign(np.diag(R))
    return Q


def _grads(w, Y, c):
    """Per-point squared gradients of f^2. Shape (n, 2k+1)."""
    f = Y @ c
    g2 = 2.0 * f[:, None] * Y
    return w * g2 * g2


def _hess(w, Y, c):
    """Positive-semidefinite Hessian approximation for sum w f^4.
    Returns (H, d2)."""
    f = Y @ c
    g = 2.0 * f[:, None] * Y          # grad f^2
    H = Y.T @ (w[:, None] * g * g)    # Gram of g, PSD
    d2 = 2.0 * f * (w * (f * f)).sum()
    return H, d2


def _ratio(Y, w, c):
    f = Y @ c
    n4 = np.sum(w * f ** 4)
    n2 = np.sum(w * f ** 2)
    if n2 <= 0:
        return np.inf
    return np.sqrt(n4) / np.sqrt(n2)


def _project(c, tol=1e-10):
    n = np.linalg.norm(c)
    if n < tol:
        return None
    return c / n


def _line_search(Y, w, c, d, f0, g0, H0, d20, alpha0, max_ls=40):
    """Damped Gauss-Newton line search along c + a*d (d is unit direction)."""
    best_c, best_f = c, f0
    best_g, best_H, best_d2 = g0, H0, d20
    alpha = alpha0
    for _ in range(max_ls):
        cand = _project(c + alpha * d)
        if cand is None:
            alpha *= 0.5
            continue
        f = Y @ cand
        n4 = np.sum(w * f ** 4)
        n2 = np.sum(w * f ** 2)
        if n2 <= 0 or not np.isfinite(n4):
            alpha *= 0.5
            continue
        val = np.sqrt(n4) / np.sqrt(n2)
        if val < best_f:
            best_f = val
            best_c = cand
            g = 2.0 * f[:, None] * Y
            best_g = w * g * g
            best_H, best_d2 = _hess(w, Y, cand)
        if val < f0 - 1e-12:
            return best_c, best_f, best_g, best_H, best_d2
        alpha *= 0.5
        if alpha < 1e-12:
            break
    return best_c, best_f, best_g, best_H, best_d2


def _solve(Y, w, c0, max_iter=250, tol=1e-12):
    c = _project(c0)
    if c is None:
        c = np.ones(Y.shape[0]) / np.sqrt(Y.shape[0])
    f = Y @ c
    n4 = np.sum(w * f ** 4)
    n2 = np.sum(w * f ** 2)
    f0 = np.sqrt(n4) / np.sqrt(n2)
    g0 = _grads(w, Y, c)
    H0, d20 = _hess(w, Y, c)
    alpha0 = 1.0
    for _ in range(max_iter):
        # gradient of log R = grad R / R
        gradR = g0 @ c / f0
        n = np.linalg.norm(gradR)
        if n < tol:
            break
        gradR = gradR / n
        # Gauss-Newton direction: solve (H - d2 I) d = -gradR
        d = None
        for lam in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0):
            A = H0 - d20 * np.eye(H0.shape[0]) + lam * np.eye(H0.shape[0])
            try:
                d = np.linalg.solve(A, -gradR)
            except np.linalg.LinAlgError:
                d = None
            if d is not None and np.all(np.isfinite(d)) and np.linalg.norm(d) > 0:
                break
        if d is None:
            d = -gradR
        d = d / np.linalg.norm(d)
        c, f0, g0, H0, d20, alpha0 = _line_search(
            Y, w, c, d, f0, g0, H0, d20, alpha0)
        alpha0 = max(alpha0, 1e-3)
    return c, f0


def rudin_problem10() -> np.ndarray:
    k = 10
    theta, phi, w = _make_grid()
    Y = _basis(k, theta, phi)
    Q = _orrows(w, Y)
    Yn = Q @ Y
    dim = 2 * k + 1

    best = None
    best_r = np.inf
    rng = np.random.default_rng(12345)
    # diverse deterministic + random orthonormal starts
    starts = [np.ones(dim) / np.sqrt(dim)]
    for _ in range(24):
        v = rng.standard_normal(dim)
        starts.append(v / np.linalg.norm(v))
    for s in starts:
        try:
            c, r = _solve(Yn, w, s)
        except Exception:
            continue
        if r < best_r:
            best_r = r
            best = c
    if best is None:
        best = np.ones(dim) / np.sqrt(dim)
    # map orthonormal-basis coefficients back to original Y basis
    c_orig = Q.T @ best
    c_orig = np.asarray(c_orig, dtype=np.float64).ravel()
    # safety: ensure validity (finite, reasonable magnitude)
    if not np.all(np.isfinite(c_orig)):
        c_orig = np.ones(dim, dtype=np.float64)
    return c_orig


# EVOLVE-BLOCK-END
