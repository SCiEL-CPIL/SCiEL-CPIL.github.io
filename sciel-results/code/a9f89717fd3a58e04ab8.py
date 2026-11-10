# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial

_START = time.time()
_LIMIT = 560.0  # safety margin before 630s kill


def _norms(m, k):
    """Normalization N_m = sqrt((2k+1)/(4pi) * (k-m)!/(k+m)!)."""
    return np.sqrt((2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m))


def _build_basis(k, ct, st, phi):
    """Return (B, w) where B is (N, 2k+1) basis values and w is per-point weight (mass 4pi)."""
    N = ct.size
    M = 2 * k + 1
    B = np.empty((N, M), dtype=np.float64)
    for m in range(-k, k + 1):
        am = abs(m)
        P = lpmv(am, k, ct)  # includes Condon-Shortley phase
        Nrm = _norms(am, k)
        if m == 0:
            B[:, am] = Nrm * P
        elif m > 0:
            B[:, am] = np.sqrt(2.0) * Nrm * P * np.cos(m * phi)
        else:
            B[:, -am] = np.sqrt(2.0) * Nrm * P * np.sin(am * phi)
    w = np.full(N, 4.0 * np.pi / N)
    return B, w


def _ratio(c, B, w):
    """R = ||f||_4 / ||f||_2 via quadrature."""
    f = B @ c
    l2 = np.sqrt(np.sum(w * f * f))
    if l2 < 1e-300:
        return np.inf
    l4 = (np.sum(w * np.abs(f) ** 4) / np.sum(w)) ** 0.25
    return l4 / l2


def _grad_l4(c, B, w):
    """Gradient of ||f||_4 (with f = B c) w.r.t. c. Returns (grad, l4)."""
    f = B @ c
    W = np.sum(w)
    l4 = (np.sum(w * np.abs(f) ** 4) / W) ** 0.25
    if l4 < 1e-300:
        return np.zeros_like(c), l4
    # d(l4)/df = (4/W) |f|^3 f / l4^3  (since l4 = (S/W)^{1/4}, S=sum w|f|^4)
    # dS/df = 4 w |f|^3 f  ->  d(l4)/df = (4/W) |f|^3 f / l4^3
    dldf = (4.0 / W) * (np.abs(f) ** 3) * f / (l4 ** 3)
    return B.T @ (w * dldf), l4


def _line_search(c, B, w, best_r):
    """Backtracking line search minimizing R along -grad direction."""
    d, _ = _grad_l4(c, B, w)
    dn = np.linalg.norm(d)
    if dn < 1e-16:
        return c, best_r, False
    d = -d / dn
    # normalize direction by typical c scale
    csc = np.linalg.norm(c)
    if csc > 1e-300:
        d = d * csc
    step = 0.5
    for _ in range(40):
        cand = c + step * d
        r = _ratio(cand, B, w)
        if r < best_r - 1e-15:
            return cand, r, True
        step *= 0.5
    return c, best_r, False


def _optimize(start, B, w, iters):
    """Gradient descent with backtracking, L2-normalized iterates."""
    c = start.copy()
    n = np.linalg.norm(c)
    if n < 1e-300:
        return c, np.inf
    c = c / n
    best_r = _ratio(c, B, w)
    best_c = c.copy()
    for _ in range(iters):
        if time.time() - _START > _LIMIT:
            break
        cand, r, improved = _line_search(c, B, w, best_r)
        if not improved:
            break
        c = cand
        n = np.linalg.norm(c)
        if n < 1e-300:
            break
        c = c / n
        if r < best_r:
            best_r = r
            best_c = c.copy()
    return best_c, best_r


def rudin_problem10() -> np.ndarray:
    k = 10
    # Quadrature: 70 Gauss-Legendre in cos(theta), 130 equally spaced phi
    ct, wt = np.polynomial.legendre.leggauss(70)  # ct in [-1,1], wt weights
    phi = 2.0 * np.pi * np.arange(130) / 130.0
    # tensor grid
    CT, PHI = np.meshgrid(ct, phi, indexing='ij')
    CT = CT.ravel()
    PHI = PHI.ravel()
    N = CT.size
    # weights: (4pi/70) * wt_i * (1/130)  -> mass = sum = 4pi
    w = np.outer(wt, np.ones(130) / 130.0).ravel() * (4.0 * np.pi / 70.0)
    B, w = _build_basis(k, CT, np.zeros(N), PHI)
    w = w  # already correct
    # rebuild with proper phi (we passed zeros; fix)
    B, w = _build_basis(k, CT, np.zeros(N), PHI)

    # candidate starting points
    starts = []
    # single modes
    for m in range(-k, k + 1):
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        starts.append(v)
    # pairs of opposite m (real even/odd combos)
    for m in range(1, k + 1):
        for s in (1.0, -1.0, 0.5, -0.5):
            v = np.zeros(2 * k + 1)
            v[m + k] = 1.0
            v[-m + k] = s
            starts.append(v)
    # random
    rng = np.random.default_rng(12345)
    for _ in range(60):
        starts.append(rng.normal(size=2 * k + 1))
    # all-ones (seed)
    starts.append(np.ones(2 * k + 1))

    best_c = np.ones(2 * k + 1)
    best_r = _ratio(best_c, B, w)

    iters = 400
    for s in starts:
        if time.time() - _START > _LIMIT:
            break
        c, r = _optimize(s, B, w, iters)
        if r < best_r:
            best_r = r
            best_c = c.copy()

    # polish: more iterations from best
    if time.time() - _START < _LIMIT - 30:
        c, r = _optimize(best_c, B, w, 800)
        if r < best_r:
            best_r = r
            best_c = c.copy()

    # scale to a safe magnitude (validity: max|c| in [1e-3, 1e5])
    n = np.linalg.norm(best_c)
    if n < 1e-300:
        best_c = np.ones(2 * k + 1)
    else:
        best_c = best_c / n
    # ensure max|c| >= 1e-3
    if np.max(np.abs(best_c)) < 1e-3:
        best_c = best_c / np.max(np.abs(best_c)) * 1.0
    if np.max(np.abs(best_c)) > 1e5:
        best_c = best_c / np.max(np.abs(best_c)) * 1.0

    return best_c.astype(np.float64)


# EVOLVE-BLOCK-END
