# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.integrate import gauss_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 600.0
_MARGIN = 30.0


def _elapsed():
    return time.time() - _START_TIME


def _deadline_ok():
    return time.time() < _DEADLINE - _MARGIN


def _build_harmonics():
    """Build 21 real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    k = 10
    n_theta = 70
    n_phi = 130

    x, w = gauss_legendre(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi

    ct = x[:, None]
    ph = phi[None, :]
    cos_phi = np.cos(ph)
    sin_phi = np.sin(ph)

    def norm_m(m):
        m = abs(int(m))
        den = 1.0
        for j in range(10 - m + 1, 10 + m + 1):
            den *= j
        return np.sqrt(21.0 / (4.0 * np.pi) / den)

    Y = np.zeros((2 * k + 1, n_theta, n_phi), dtype=np.float64)
    for m in range(-k, k + 1):
        i = m + k
        if m == 0:
            P0 = lpmv(0, k, ct)
            N0 = norm_m(0)
            Y[i, :, :] = N0 * P0
        elif m > 0:
            Pm = lpmv(m, k, ct)
            Nm = norm_m(m)
            Y[i, :, :] = np.sqrt(2.0) * Nm * Pm * cos_phi
        else:
            am = -m
            Pam = lpmv(am, k, ct)
            Nam = norm_m(am)
            Y[i, :, :] = np.sqrt(2.0) * Nam * Pam * sin_phi
    return Y, x, w, phi, n_theta, n_phi


def _build_data(Y, x, w, phi, n_theta, n_phi):
    dphi = 2.0 * np.pi / n_phi
    cell_w = (w[:, None] * dphi * np.ones((1, n_phi), dtype=np.float64)).ravel()
    Yf = Y.reshape(21, -1)
    return Yf, cell_w


def _obj_grad(c, Yf, W):
    """Return (||f||_4^4, gradient) for f = Yf @ c."""
    f = Yf @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    obj = np.dot(W, f4)
    grad = 4.0 * (Yf.T @ (W * f3))
    return obj, grad


def _proj_sphere(c):
    n = np.linalg.norm(c)
    if n < 1e-15:
        return c
    return c / n


def _pjd(c0, Yf, W, max_iter=4000, lr=0.004, tol=1e-13):
    """Projected gradient descent with backtracking line search on the sphere."""
    c = _proj_sphere(c0.astype(np.float64).copy())
    obj, grad = _obj_grad(c, Yf, W)
    best_c = c.copy()
    best_obj = obj
    for it in range(max_iter):
        if not _deadline_ok():
            break
        # Backtracking line search
        step = lr
        c_new = None
        obj_new = None
        for _ in range(30):
            cand = c - step * grad
            cand = _proj_sphere(cand)
            obj_c, _ = _obj_grad(cand, Yf, W)
            if obj_c < obj - 1e-4 * step * np.dot(grad, c - cand):
                c_new = cand
                obj_new = obj_c
                break
            step *= 0.5
        if c_new is None:
            # Take a small step anyway
            c_new = _proj_sphere(c - 1e-6 * grad)
            obj_new, _ = _obj_grad(c_new, Yf, W)
        if obj_new < best_obj:
            best_obj = obj_new
            best_c = c_new.copy()
        if abs(obj_new - obj) < tol * max(1.0, abs(obj)):
            break
        c, obj = c_new, obj_new
        _, grad = _obj_grad(c, Yf, W)
    return best_c, best_obj


def rudin_problem10() -> np.ndarray:
    k = 10
    Y, x, w, phi, n_theta, n_phi = _build_harmonics()
    Yf, W = _build_data(Y, x, w, phi, n_theta, n_phi)

    # Seed: all-ones
    best_c = np.ones(2 * k + 1, dtype=np.float64)
    best_c = _proj_sphere(best_c)
    best_obj, _ = _obj_grad(best_c, Yf, W)

    rng = np.random.default_rng(12345)
    n_restarts = 60

    for r in range(n_restarts):
        if not _deadline_ok():
            break

        if r == 0:
            c0 = best_c.copy()
        elif r % 5 == 1:
            c0 = rng.standard_normal(2 * k + 1)
        elif r % 5 == 2:
            c0 = np.zeros(2 * k + 1)
            idx = rng.choice(2 * k + 1, size=int(rng.integers(4, 12)), replace=False)
            c0[idx] = rng.standard_normal(len(idx))
        elif r % 5 == 3:
            # Emphasize m=0 and a few others
            c0 = np.zeros(2 * k + 1)
            c0[10] = 1.0
            for _ in range(int(rng.integers(2, 6))):
                c0[int(rng.integers(0, 21))] += rng.uniform(0.2, 0.8)
        else:
            # Gaussian mixture-like
            c0 = rng.standard_normal(2 * k + 1) * rng.uniform(0.1, 1.0)
            c0[int(rng.integers(0, 21))] += rng.uniform(1.0, 3.0)

        c0 = _proj_sphere(c0)
        c_new, obj_new = _pjd(c0, Yf, W, max_iter=2500, lr=0.004)
        if obj_new < best_obj:
            best_obj = obj_new
            best_c = c_new.copy()

    # Final polish
    if _deadline_ok():
        c_pol, obj_pol = _pjd(best_c, Yf, W, max_iter=8000, lr=0.001, tol=1e-14)
        if obj_pol < best_obj:
            best_obj = obj_pol
            best_c = c_pol.copy()

    best_c = np.asarray(best_c, dtype=np.float64)
    if best_c.shape != (21,) or not np.all(np.isfinite(best_c)):
        best_c = np.ones(21, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3 or mx > 1e5:
        best_c = np.ones(21, dtype=np.float64)

    return best_c


# EVOLVE-BLOCK-END
