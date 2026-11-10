# EVOLVE-BLOCK-START
import time
import numpy as np

_K = 10
_NPT = 70   # Gauss-Legendre nodes in cos(theta)
_NPH = 130  # equally spaced phi


def _assoc_legendre(n, m, x):
    """P_n^m(x) with Condon-Shortley phase, for |x|<=1 (stable upward recursion)."""
    m = abs(m)
    x = np.asarray(x, dtype=np.float64)
    Pmm = np.ones_like(x)
    if m > 0:
        sq = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        for i in range(1, m + 1):
            Pmm *= -i * sq
    if n == m:
        return Pmm
    Pmp1 = x * (2 * m + 1) * Pmm
    if n == m + 1:
        return Pmp1
    Pm2 = Pmm
    Pm1 = Pmp1
    for l in range(m + 2, n + 1):
        Pm2, Pm1 = Pm1, (x * (2 * l - 1) * Pm1 - (l + m - 1) * Pm2) / (l - m)
    return Pm1


def _build_basis():
    """Return (Y, w) where Y has shape (N, 21) columns = real orthonormal Y_m
    (m = -10..10) and w is the per-node quadrature weight (sums to 4*pi)."""
    x, w = np.polynomial.legendre.leggauss(_NPT)          # x = cos(theta) in [-1,1]
    phi = 2.0 * np.pi * np.arange(_NPH) / _NPH
    Y = np.empty((_NPT, _NPH, 2 * _K + 1), dtype=np.float64)
    for m in range(0, _K + 1):
        Pm = _assoc_legendre(_K, m, x)                    # (NPT,)
        N = np.sqrt((2 * _K + 1) / (4.0 * np.pi) *
                    float(np.math.factorial(_K - m) / np.math.factorial(_K + m)))
        Y[:, :, m] = N * Pm[:, None]
        if m > 0:
            Y[:, :, -m] = np.sqrt(2.0) * N * Pm[:, None] * np.cos(m * phi)[None, :]
            Y[:, :, m] = np.sqrt(2.0) * N * Pm[:, None] * np.sin(m * phi)[None, :]
    Y = Y.reshape(-1, 2 * _K + 1)
    w = np.kron(w, np.full(_NPH, 1.0 / _NPH))             # sums to 4*pi
    return Y, w


def _norms(c, Y, w):
    f = Y @ c
    l2sq = float(np.dot(w, f * f))
    l4 = float(np.dot(w, f ** 4))
    return l2sq, l4


def _fixed_point(c, Y, w, iters=600, tol=1e-14):
    """Fixed-point iteration c <- Proj(f^3)/||f||_2 (critical-point search)."""
    c = np.array(c, dtype=np.float64)
    c /= np.linalg.norm(c)
    for _ in range(iters):
        f = Y @ c
        l2sq = float(np.dot(w, f * f))
        g = Y.T @ (f ** 3 * w)
        ng = np.linalg.norm(g)
        if ng < 1e-300:
            break
        c_new = g / ng
        d = np.abs(c_new - c)
        c = c_new
        if d.max() < tol:
            break
    return c


def _newton_polish(c, Y, w, maxit=40, tol=1e-15):
    """Projected Newton on the unit sphere for min J = int f^4 (with ||f||_2=1)."""
    c = np.array(c, dtype=np.float64)
    c /= np.linalg.norm(c)
    n = c.size
    for _ in range(maxit):
        f = Y @ c
        l2sq = float(np.dot(w, f * f))
        J = float(np.dot(w, f ** 4))
        # gradient of J on the sphere (J homogeneous quartic)
        grad = 4.0 * Y.T @ (f ** 3 * w)
        # Hessian of J
        H = 12.0 * (Y.T @ ((f ** 2 * w)[:, None] * Y))
        lam = 2.0 * J / l2sq
        rhs = grad - lam * c
        A = H - lam * np.eye(n)
        # solve in tangent space
        try:
            Ainv_r = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            Ainv_r = np.linalg.lstsq(A, rhs, rcond=None)[0]
        d = Ainv_r - (Ainv_r @ c) * c
        Jd = float(np.dot(w, (f + Y @ d) ** 4))
        step = 1.0
        improved = False
        for _b in range(60):
            c_try = c + step * d
            c_try /= np.linalg.norm(c_try)
            Jt = float(np.dot(w, (Y @ c_try) ** 4))
            if Jt < J:
                c = c_try
                improved = True
                break
            step *= 0.5
        if not improved:
            break
        if np.abs(Jd - J) < tol * max(1.0, J):
            break
    return c


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0
    Y, w = _build_basis()
    n = 2 * _K + 1

    def R_of(c):
        l2sq, l4 = _norms(c, Y, w)
        return (l4 / l2sq) ** 0.25

    # candidate starts
    starts = [np.ones(n)]
    rng = np.random.default_rng(20240517)
    for _ in range(24):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        starts.append(v)
    # a few structured starts (single harmonics, alternating signs)
    for m in (-_K, -5, 0, 5, _K):
        e = np.zeros(n)
        e[m + _K] = 1.0
        starts.append(e)
    for s in (1, -1):
        a = np.zeros(n)
        for i in range(n):
            a[i] = s ** (i % 2)
        starts.append(a)

    best = None
    bestR = np.inf
    for c0 in starts:
        if time.time() > deadline - 30.0:
            break
        c = _fixed_point(c0, Y, w)
        c = _newton_polish(c, Y, w)
        R = R_of(c)
        if R < bestR:
            bestR = R
            best = c.copy()

    # safety: ensure valid magnitude (scale-invariant; keep unit norm)
    if best is None:
        best = np.ones(n)
    best = best / np.linalg.norm(best)
    if not np.all(np.isfinite(best)):
        best = np.ones(n)
    return best


# EVOLVE-BLOCK-END
