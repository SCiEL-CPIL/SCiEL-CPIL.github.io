# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

_DEADLINE = 600.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_grid():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-equal phi grid
    and the (N, 21) real orthonormal degree-10 spherical-harmonic matrix Y."""
    x, w = np.polynomial.legendre.leggauss(_NTH)          # nodes in [-1,1]
    # weight for the sphere: dOmega = dphi d(cos theta); phi weight = 2pi/NPHI
    wphi = 2.0 * np.pi / _NPHI
    wt = np.repeat(w, _NPHI) * wphi                        # length NTH*NPHI, sums to 4pi
    phi = np.arange(_NPHI) * (2.0 * np.pi / _NPHI)         # equally spaced
    cth = np.repeat(x, _NPHI)                              # cos theta, length N
    sth = np.sqrt(np.clip(1.0 - cth * cth, 0.0, 1.0))      # sin theta
    phirep = np.tile(phi, _NTH)                            # phi, length N

    N = _NTH * _NPHI
    Y = np.empty((N, 2 * _K + 1), dtype=np.float64)
    for m in range(0, _K + 1):
        norm = np.sqrt(2 * _K + 1 / (4.0 * np.pi) *
                       float(np.math.factorial(_K - m)) /
                       float(np.math.factorial(_K + m)))
        P = lpmv(m, _K, cth)                               # includes Condon-Shortley (-1)^m
        if m == 0:
            Y[:, m] = norm * P
        else:
            Y[:, m] = np.sqrt(2.0) * norm * P * np.cos(m * phirep)
            Y[:, _K - m] = np.sqrt(2.0) * norm * P * np.sin(m * phirep)
    return Y, wt


def _norms(Y, wt, c):
    f = Y @ c
    l2 = float(np.dot(wt, f * f))
    l4 = float(np.dot(wt, f ** 4))
    return l2, l4


def _objective(c, Y, wt):
    l2, l4 = _norms(Y, wt, c)
    if l2 <= 0.0:
        return 1e12
    return l4 / l2


def rudin_problem10() -> np.ndarray:
    """Return 21 real coefficients (m = -10..10) minimizing the L4/L2 ratio
    of the degree-10 spherical harmonic on the unit sphere.

    The objective is scale-invariant, so we fix ||f||_2 = 1 and minimize ||f||_4.
    Uses the exact 70x130 quadrature grid; L-BFGS-B with random restarts.
    """
    t0 = time.time()
    Y, wt = _build_grid()

    # orthonormal basis vectors (each has ||.||_2 = 1 exactly)
    basis = np.linalg.inv(Y.T @ (wt[:, None] * Y))[:, :2 * _K + 1]
    # basis[:, j] is a unit-L2 coefficient vector
    # objective for a unit-L2 vector is simply ||f||_4 = sum wt f^4
    def obj_unit(c):
        f = Y @ c
        return float(np.dot(wt, f ** 4))

    def grad_unit(c):
        f = Y @ c
        return 4.0 * Y.T @ (wt * f ** 3)

    best_c = None
    best_val = np.inf

    def time_left():
        return _DEADLINE - 60.0 - (time.time() - t0)

    rng = np.random.default_rng(12345)
    # number of restarts; each L-BFGS-B is fast (9100x21 matrix)
    n_restarts = 4000

    for it in range(n_restarts):
        if time_left() < 1.0:
            break
        # random unit-L2 start: random direction in coefficient space, then
        # project to unit L2 norm
        v = rng.standard_normal(2 * _K + 1)
        l2 = float(np.dot(wt, (Y @ v) ** 2))
        if l2 <= 0:
            continue
        c0 = v / np.sqrt(l2)
        try:
            res = minimize(obj_unit, c0, jac=grad_unit, method='L-BFGS-B',
                           options={'maxiter': 200, 'gtol': 1e-9,
                                    'ftol': 1e-12, 'maxls': 40})
        except Exception:
            continue
        c = res.x
        val = obj_unit(c)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    if best_c is None:
        # fallback: all-ones (valid, non-degenerate)
        best_c = np.ones(2 * _K + 1, dtype=np.float64)

    # sanity: ensure finite and within magnitude bounds
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1, dtype=np.float64)
    mmax = float(np.max(np.abs(best_c)))
    if mmax < 1e-3:
        best_c = best_c * (1.0 / max(mmax, 1e-12))
    if mmax > 1e5:
        best_c = best_c * (1.0 / mmax)
    return best_c.astype(np.float64)


# EVOLVE-BLOCK-END
