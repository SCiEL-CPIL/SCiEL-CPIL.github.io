# EVOLVE-BLOCK-START
import numpy as np


def _norms(n, m):
    """N_m = sqrt( (2n+1)/(4 pi) * (n-m)!/(n+m)! ) for |m| <= n."""
    m = abs(int(m))
    return np.sqrt((2 * n + 1) / (4.0 * np.pi) * float(np.math.factorial(n - m)) /
                   float(np.math.factorial(n + m)))


def _real_harmonics(n, cos_t, sin_t, cos_phi, sin_phi):
    """
    Build the real orthonormal degree-n spherical harmonics evaluated at
    (cos_t, phi) = (cos_t, atan2(sin_phi, cos_phi)).
    Returns Y of shape (2n+1, N) with Y[i] for m = i - n:
        m =  0 : Y_0  = N_0 * P_n(cos_t)
        m > 0 : Y_m  = sqrt(2) N_m P_n^m(cos_t) cos(m phi)
        m < 0 : Y_m  = sqrt(2) N_|m| P_n^|m|(cos_t) sin(|m| phi)
    """
    Npts = cos_t.shape[0]
    Y = np.zeros((2 * n + 1, Npts), dtype=np.float64)
    for m in range(0, n + 1):
        P = np.polynomial.legendre.legval(cos_t,
                                          np.polynomial.legendre.leggauss(1)[0] and np.zeros(1))  # placeholder
        # Use scipy-free recurrence: P_n^m via associated Legendre (Condon-Shortley)
        Pm = _assoc_legendre(n, m, cos_t)
        Nm = _norms(n, m)
        idx = n + m
        if m == 0:
            Y[idx, :] = Nm * Pm
        else:
            Y[idx, :] = np.sqrt(2.0) * Nm * Pm * cos_phi
            Y[n - m, :] = np.sqrt(2.0) * Nm * Pm * sin_phi
    return Y


def _assoc_legendre(n, m, x):
    """Associated Legendre P_n^m(x) with Condon-Shortley phase, m >= 0, n >= m."""
    x = np.asarray(x, dtype=np.float64)
    # P_m^m
    Pmm = np.ones_like(x)
    if m > 0:
        fact = 1.0
        sq = np.sqrt((1.0 - x) * (1.0 + x))
        for i in range(1, m + 1):
            Pmm *= -fact * sq
            fact += 2.0
    if n == m:
        return Pmm
    Pmp1 = x * (2 * m + 1) * Pmm
    if n == m + 1:
        return Pmp1
    Pmm_prev = Pmm
    Pm = Pmp1
    for i in range(m + 2, n + 1):
        Pmm_prev, Pm = Pm, ((2 * i - 1) * x * Pm - (i + m - 1) * Pmm_prev) / (i - m)
    return Pm


def _build_quadrature(n, n_theta=70, n_phi=130):
    """Gauss-Legendre in cos(theta) x equally spaced phi. Returns weights, cos_t, phi."""
    x, w = np.polynomial.legendre.leggauss(n_theta)  # x in [-1,1], weights for integral over [-1,1]
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    # Mesh: index = i_theta * n_phi + i_phi
    cos_t = np.repeat(x, n_phi)
    phi_v = np.tile(phi, n_theta)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    sin_t = np.repeat(sin_t, n_phi)
    # weight per point: (dOmega = dphi dcos) -> w_theta * (2pi/n_phi)
    wpts = np.repeat(w * (2.0 * np.pi / n_phi), n_theta)
    return wpts, cos_t, sin_t, phi_v


def _make_Y(wpts, cos_t, sin_t, phi_v, n=10):
    cos_phi = np.cos(phi_v)
    sin_phi = np.sin(phi_v)
    return _real_harmonics(n, cos_t, sin_t, cos_phi, sin_phi)


def _evaluate(c, Y, wpts):
    """Return (L4_over_L2, grad) for f = sum c_m Y_m. Norms over sphere, total mass 4pi."""
    f = Y.T @ c  # (N,)
    f2 = f * f
    f4 = f2 * f2
    mass = wpts.sum()  # = 4 pi
    L2sq = (wpts * f2).sum() / mass
    L4 = (wpts * f4).sum() / mass
    L2 = np.sqrt(max(L2sq, 1e-300))
    L4v = L4 ** 0.25
    ratio = L4v / L2
    # gradient of ratio
    if L2 <= 1e-300:
        return ratio, np.zeros_like(c)
    # d(ratio)/dc = (1/L2) * (1/L4v) * (1/4) * 4 f^3 * f' - (L4v/L2^2) * (1/L2sq) * 2 f f'
    # = (f^3 / (L2 * L4v)) * f' - (L4v * f / (L2^2 * L2sq)) * f'
    fp = Y  # d f / d c_m = Y_m
    g = (f ** 3 / (L2 * L4v) - L4v * f / (L2 * L2 * L2sq)) * fp  # (N,)
    grad = Y @ g  # (2n+1,)
    return ratio, grad


def _optimize(n=10, n_theta=70, n_phi=130, max_iter=4000, tol=1e-12):
    wpts, cos_t, sin_t, phi_v = _build_quadrature(n, n_theta, n_phi)
    Y = _make_Y(wpts, cos_t, sin_t, phi_v, n)
    dim = 2 * n + 1
    # Start from a non-degenerate seed: a single dominant m plus small others
    rng = np.random.default_rng(0)
    best_c = None
    best_val = np.inf
    starts = []
    # seed 0: all ones (matches parent, guarantees validity)
    starts.append(np.ones(dim, dtype=np.float64) / np.sqrt(dim))
    # seed 1: dominant m=0
    a = np.zeros(dim); a[n] = 1.0; starts.append(a)
    # seed 2: dominant m=10 (cos)
    b = np.zeros(dim); b[2 * n] = 1.0; starts.append(b)
    # seed 3: dominant m=-10 (sin)
    d = np.zeros(dim); d[0] = 1.0; starts.append(d)
    # seed 4: random
    r = rng.standard_normal(dim); r /= np.linalg.norm(r); starts.append(r)
    # seed 5: random
    r2 = rng.standard_normal(dim); r2 /= np.linalg.norm(r2); starts.append(r2)

    for c0 in starts:
        c0 = c0 / np.linalg.norm(c0)
        # L-BFGS-B on ratio(c)
        try:
            from scipy.optimize import minimize
            res = minimize(_evaluate, c0, args=(Y, wpts), method='L-BFGS-B',
                           jac=True, options={'maxiter': max_iter, 'ftol': tol,
                                              'gtol': 1e-10, 'maxls': 50})
            c = res.x
            val, _ = _evaluate(c, Y, wpts)
        except Exception:
            val = np.inf
        # enforce unit norm (ratio is scale-invariant, keeps max|c| in range)
        cn = np.linalg.norm(c)
        if cn > 0:
            c = c / cn
        val, _ = _evaluate(c, Y, wpts)
        if val < best_val:
            best_val = val
            best_c = c.copy()
    if best_c is None:
        best_c = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    # Scale so max|c| is comfortably within (1e-3, 1e5): keep unit norm (max ~1)
    return best_c


def rudin_problem10() -> np.ndarray:
    """
    Return 21 real coefficients c_m (m = -10..10) for a degree-10 spherical
    harmonic f = sum_m c_m Y_m that minimizes the L4/L2 norm ratio on S^2.

    The returned vector is unit-norm, so 1e-3 < max|c| < 1e5 holds.
    """
    try:
        c = _optimize(n=10, n_theta=70, n_phi=130)
    except Exception:
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    if c.shape[0] != 21:
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    if not np.all(np.isfinite(c)):
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    # ensure max|c| in valid range
    mx = np.max(np.abs(c))
    if mx < 1e-3:
        c = c / max(mx, 1e-12) * 0.5
    elif mx > 1e5:
        c = c / mx * 0.5
    return c


# EVOLVE-BLOCK-END
