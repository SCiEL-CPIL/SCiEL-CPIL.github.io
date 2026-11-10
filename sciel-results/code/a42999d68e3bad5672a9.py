# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0
_MARGIN = 20.0


def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    mu, w = np.polynomial.legendre.leggauss(70)
    mu = mu[::-1]
    w = w[::-1]
    nphi = 130
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    dphi = 2.0 * np.pi / nphi
    return mu, w, phi, dphi


def _assoc_leg(k, m, x):
    """Associated Legendre P_k^m(x) with Condon-Shortley phase, via scipy."""
    from scipy.special import lpmv
    return lpmv(m, k, x)


def _build_basis(k, mu, phi):
    """Return (Y, wq) where Y[i, j] = Y_i at grid point j, i in 0..2k (m=i-k)."""
    nphi = phi.size
    nm = 2 * k + 1
    Y = np.empty((nm, mu.size * nphi), dtype=np.float64)
    for i in range(nm):
        m = i - k
        am = abs(m)
        # N_m = sqrt(21/(4 pi) * (k-am)!/(k+am)!)  for k=10 -> 21/(4pi)
        N = np.sqrt(21.0 / (4.0 * np.pi) *
                    float(np.math.factorial(k - am)) /
                    float(np.math.factorial(k + am)))
        P = _assoc_leg(k, am, mu)  # shape (nmu,)
        if m == 0:
            vals = N * P[:, None] * np.ones((1, nphi))
        elif m > 0:
            vals = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            vals = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]
        Y[i] = vals.ravel()
    # quadrature weights: w_mu * dphi
    wq = np.repeat(w * dphi, nphi)
    return Y, wq


def _ratios(coeffs, Y, wq):
    """Return (R, f2) where f = Y.T @ coeffs, R = ||f||4/||f||2."""
    f = Y.T @ coeffs
    f2 = np.sum(wq * f * f)
    if f2 <= 1e-300:
        return np.inf, f2
    f4 = np.sum(wq * f**4)
    n4 = f4 ** 0.25
    n2 = f2 ** 0.5
    return n4 / n2, f2


def _flattening_step(coeffs, Y, wq, omega=1.0):
    """Fixed point: c <- omega * (Y^T w f^3)/||...|| + (1-omega) c, L2-normalized."""
    f = Y.T @ coeffs
    f2 = np.sum(wq * f * f)
    if f2 <= 1e-300:
        return coeffs
    target = Y.T @ (wq * f**3)
    norm_t = np.sqrt(np.sum(target * target))
    if norm_t <= 1e-300:
        return coeffs
    newc = omega * target / norm_t + (1.0 - omega) * (coeffs / np.sqrt(np.sum(coeffs**2)))
    nn = np.sqrt(np.sum(newc**2))
    if nn <= 1e-300:
        return coeffs
    return newc / nn


def _local_polish(coeffs, Y, wq, max_iter=400):
    """Line search + flattened fixed point to reduce R."""
    coeffs = coeffs / np.sqrt(np.sum(coeffs**2))
    R, _ = _ratios(coeffs, Y, wq)
    best = coeffs.copy()
    bestR = R
    for it in range(max_iter):
        c = _flattening_step(best, Y, wq, omega=1.0)
        # line search on alpha: best -> c
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            cand = (1.0 - alpha) * best + alpha * c
            n = np.sqrt(np.sum(cand**2))
            if n <= 1e-300:
                continue
            cand = cand / n
            Rc, _ = _ratios(cand, Y, wq)
            if Rc < bestR - 1e-12:
                best = cand
                bestR = Rc
                improved = True
                break
        if not improved:
            break
    return best, bestR


def _gen_starts(k):
    starts = []
    rng = np.random.default_rng(12345)
    # all ones
    starts.append(np.ones(2 * k + 1))
    # only m=0
    for m0 in (0, 1, 2, 5, 10):
        v = np.zeros(2 * k + 1)
        v[m0 + k] = 1.0
        starts.append(v)
    # random
    for _ in range(24):
        starts.append(rng.standard_normal(2 * k + 1))
    # sign-symmetric patterns
    for _ in range(8):
        v = np.zeros(2 * k + 1)
        for m in range(k + 1):
            v[m + k] = 1.0
            v[k - m] = 1.0 if m == 0 else (1.0 if rng.random() < 0.5 else -1.0)
        starts.append(v)
    return starts


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    k = 10
    mu, w, phi, dphi = _build_quadrature()
    Y, wq = _build_basis(k, mu, phi)

    starts = _gen_starts(k)
    best = starts[0].copy()
    bestR = np.inf
    for s in starts:
        if time.time() - t0 > _DEADLINE - _MARGIN:
            break
        c, R = _local_polish(s, Y, wq, max_iter=200)
        if R < bestR:
            bestR = R
            best = c

    # final high-precision polish
    if time.time() - t0 < _DEADLINE - _MARGIN:
        best, bestR = _local_polish(best, Y, wq, max_iter=800)

    # scale: any nonzero scale is valid; use moderate magnitude
    n = np.sqrt(np.sum(best**2))
    if n <= 1e-300:
        best = np.ones(2 * k + 1)
    else:
        best = best / n
        # scale up so max|c| is in a safe, informative range
        best = best * 1.0

    if not np.all(np.isfinite(best)) or best.size != 2 * k + 1:
        return np.ones(2 * k + 1, dtype=np.float64)
    return best.astype(np.float64)


# EVOLVE-BLOCK-END
