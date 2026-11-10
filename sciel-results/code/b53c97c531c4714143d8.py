# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 600.0
_MARGIN = 15.0

# ---- Precomputation (module-level, one-time) ----
_K = 10
_NTH = 70
_NPHI = 130

def _build_basis():
    # Gauss-Legendre nodes/weights on [-1,1]
    x, w = np.polynomial.legendre.leggauss(_NTH)
    # quadrature weight on sphere: 2*pi*w (phi integral of 1 over [0,2pi])
    wq = 2.0 * np.pi * w
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # Associated Legendre P_l^m(cos theta) with Condon-Shortley phase.
    # Use scipy if available for exactness; else implement.
    try:
        from scipy.special import lpmv
        P = lpmv(np.arange(_K + 1)[None, :], _K, x[:, None])  # shape (NTH, K+1)
    except Exception:
        # manual associated Legendre with Condon-Shortley
        P = np.zeros((_NTH, _K + 1))
        for m in range(_K + 1):
            for i in range(_NTH):
                P[i, m] = _assoc_legendre(_K, m, x[i])

    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    N = np.zeros(_K + 1)
    for m in range(_K + 1):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * _factorial_ratio(_K - m, _K + m))

    # Build basis matrix B of shape (NTH, NPHI, 2K+1)
    B = np.zeros((_NTH, _NPHI, 2 * _K + 1))
    for m in range(_K + 1):
        Nm = N[m]
        if m == 0:
            Y0 = Nm * P[:, 0][:, None]
            B[:, :, 10] = Y0
        else:
            Yc = np.sqrt(2.0) * Nm * P[:, m][:, None] * np.cos(m * phi)[None, :]
            Ys = np.sqrt(2.0) * Nm * P[:, m][:, None] * np.sin(m * phi)[None, :]
            B[:, :, 10 + m] = Yc
            B[:, :, 10 - m] = Ys

    # Flatten spatial axes
    Bf = B.reshape(_NTH * _NPHI, 2 * _K + 1)
    # Full quadrature weights (wq * dphi)
    W = np.repeat(wq, _NPHI) * dphi
    return Bf, W

def _assoc_legendre(l, m, x):
    # P_l^m with Condon-Shortley phase, x = cos theta
    if m < 0:
        return (-1) ** m * np.math.factorial(m) / np.math.factorial(-m) * _assoc_legendre(l, -m, x)
    # start from P_m^m
    Pmm = 1.0
    if m > 0:
        s = 1.0
        for i in range(1, m + 1):
            s *= -np.sqrt((1.0 - x) * (1.0 + x))
        Pmm = s
    if l == m:
        return Pmm
    Pm1 = x * (2 * m + 1) * Pmm
    if l == m + 1:
        return Pm1
    Pm2 = Pmm
    for ll in range(m + 2, l + 1):
        Pm2 = ((2 * ll - 1) * x * Pm1 - (ll + m - 1) * Pm2) / (ll - m)
        Pm1 = Pm2
    return Pm1

def _factorial_ratio(a, b):
    # a! / b!  (a <= b)
    r = 1.0
    for i in range(a + 1, b + 1):
        r /= i
    return r

_BF, _W = _build_basis()
_W2 = _W ** 2

def _compute_R(c):
    f = _BF.dot(c)
    n4 = float(np.sum(_W * f ** 4))
    n2 = float(np.sum(_W * f * f))
    if n2 <= 0.0:
        return np.inf
    return (n4 / n2) ** 0.5

def _compute_R_grad(c):
    f = _BF.dot(c)
    n4 = float(np.sum(_W * f ** 4))
    n2 = float(np.sum(_W * f * f))
    if n2 <= 0.0:
        return np.inf, np.zeros_like(c)
    R = (n4 / n2) ** 0.5
    # dR/dc = (1/(2 R)) * (4*n2*sum(w f^3 B) - n4*sum(2 w f B)/n2)
    df4 = _W * f ** 3
    df2 = _W * f
    g = (2.0 * n2 * _BF.T.dot(df4) - n4 * _BF.T.dot(df2) / n2)
    g *= (1.0 / (2.0 * R))
    return R, g

def _line_search(c, R, g, alpha):
    # backtracking on R
    step = -g
    norm_step = np.linalg.norm(step)
    if norm_step == 0.0:
        return c, R
    step = step / norm_step
    # ensure descent direction
    if np.dot(g, step) > 0:
        step = -step
    a = alpha
    R_new = R
    c_new = c
    for _ in range(40):
        cand = c + a * step
        R_cand = _compute_R(cand)
        if R_cand < R_new:
            R_new = R_cand
            c_new = cand
            break
        a *= 0.5
    return c_new, R_new

def _optimize_from(c0, iters, alpha):
    c = c0.copy()
    R = _compute_R(c)
    for it in range(iters):
        if time.time() > _DEADLINE - _MARGIN:
            break
        R, g = _compute_R_grad(c)
        c, R = _line_search(c, R, g, alpha)
    return c, R

def rudin_problem10() -> np.ndarray:
    rng = np.random.default_rng(20240517)
    best_c = np.ones(2 * _K + 1, dtype=np.float64)
    best_R = _compute_R(best_c)

    # Restart 1: all-ones
    c, R = _optimize_from(best_c.copy(), 400, 1.0)
    if R < best_R:
        best_c, best_R = c, R

    # Random restarts
    for restart in range(40):
        if time.time() > _DEADLINE - _MARGIN:
            break
        c0 = rng.normal(size=2 * _K + 1)
        c0 /= (np.linalg.norm(c0) + 1e-15)
        c, R = _optimize_from(c0, 250, 1.0)
        if R < best_R:
            best_R = R
            best_c = c
            # polish from improved point
            if time.time() < _DEADLINE - _MARGIN:
                c2, R2 = _optimize_from(best_c.copy(), 300, 0.5)
                if R2 < best_R:
                    best_R, best_c = R2, c2

    # final validation
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1)
    return best_c.astype(np.float64)
# EVOLVE-BLOCK-END
