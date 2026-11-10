# EVOLVE-BLOCK-START
import time
import numpy as np

_K = 10
_NTHETA = 70
_NPHI = 130
_DEADLINE = 600.0  # safety margin vs 630s kill


def _legendre_p_lm(l, m, x):
    """Associated Legendre P_l^m(x) with Condon-Shortley phase, via recurrence."""
    # Compute P_m^m first
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = -pmm * i * somx2
            fact *= i
    # pmm already includes (-1)^m * prod, but standard recurrence:
    # P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    # Let's recompute cleanly:
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(0.0, (1.0 - x) * (1.0 + x)))
        for i in range(1, m + 1):
            pmm = -pmm * i * somx2
    if m == l:
        return pmm
    # P_{m+1}^m
    pmmp1 = (2 * m + 1) * x * pmm
    if l == m + 1:
        return pmmp1
    pll = pmmp1
    for ll in range(m + 2, l + 1):
        pmm, pmmp1 = pmmp1, ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pll = pmmp1
    return pll


def _build_basis():
    """Return (basis, wtheta) where basis has shape (N, 21), N = NTHETA*NPHI,
    ordered m=-10..10. wtheta is the Gauss-Legendre weight (per theta node)
    such that sum over grid with wtheta * (1/NPHI) approximates surface integral
    (dOmega = dphi dcos theta)."""
    # Gauss-Legendre nodes/weights on [-1,1]
    x, w = np.polynomial.legendre.leggauss(_NTHETA)
    # phi grid
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    # Build full grid
    theta_c = x  # cos theta
    theta_c_grid, phi_grid = np.meshgrid(theta_c, phi, indexing='ij')
    theta_c_flat = theta_c_grid.ravel()
    phi_flat = phi_grid.ravel()
    N = theta_c_flat.size

    basis = np.empty((N, 2 * _K + 1), dtype=np.float64)
    for m in range(-_K, _K + 1):
        am = abs(m)
        # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        # log for stability
        from math import lgamma, log, pi
        logN = 0.5 * (log(2 * _K + 1) - log(4 * pi) + lgamma(_K - am + 1) - lgamma(_K + am + 1))
        Nm = np.exp(logN)
        Plm = _legendre_p_lm(_K, am, theta_c_flat)
        if m == 0:
            basis[:, m + _K] = Nm * Plm
        elif m > 0:
            basis[:, m + _K] = np.sqrt(2.0) * Nm * Plm * np.cos(m * phi_flat)
        else:
            basis[:, m + _K] = np.sqrt(2.0) * Nm * Plm * np.sin(am * phi_flat)

    # Quadrature weight per grid point: (wtheta_i / NTHETA) * (2pi / NPHI)
    # Actually surface measure dOmega = dphi dcos theta. Gauss-Legendre on [-1,1]
    # approximates integral over cos theta with weights w (sum w_i f(x_i) ~ int_{-1}^{1}).
    # phi integral over [0,2pi] with uniform spacing: (2pi/NPHI) * sum.
    # So per-point weight = w_i * (2pi/NPHI).
    wtheta = w * (2.0 * np.pi / _NPHI)
    # Flatten weight: each theta node repeats NPHI times
    w_flat = np.repeat(wtheta, _NPHI)
    return basis, w_flat


def _compute_R2(f, w, basis):
    """Compute R^2 = (sum w f^4)/(sum w f^2) using precomputed f = basis @ c? 
    Actually we pass f directly. Returns R^2 and f^2, f^4 sums."""
    f2 = f * f
    s2 = np.dot(w, f2)
    s4 = np.dot(w, f2 * f2)
    if s2 <= 0:
        return np.inf, s2, s4
    return s4 / s2, s2, s4


def _compute_R(f, w):
    R2, s2, s4 = _compute_R2(f, w)
    return np.sqrt(R2)


def _grad_R2(c, basis, w):
    """Gradient of R^2 = S4/S2 w.r.t. c, where f = basis @ c."""
    f = basis @ c
    f2 = f * f
    s2 = np.dot(w, f2)
    s4 = np.dot(w, f2 * f2)
    if s2 <= 0:
        return np.zeros_like(c)
    # dR2/dc = (S2 * dS4 - S4 * dS2)/S2^2
    # dS4/dc = 4 * w * f^3 (vector) -> dot with basis
    # dS2/dc = 2 * w * f -> dot with basis
    dS4 = 4.0 * basis.T @ (w * f * f2)
    dS2 = 2.0 * basis.T @ (w * f)
    return (s2 * dS4 - s4 * dS2) / (s2 * s2)


def _adam(c, basis, w, lr0=0.05, iters=2000, beta1=0.9, beta2=0.999, eps=1e-8,
          deadline=None):
    """Adam minimization of R^2."""
    c = c.copy()
    m = np.zeros_like(c)
    v = np.zeros_like(c)
    best_c = c.copy()
    best_R = _compute_R(c, w)
    for t in range(1, iters + 1):
        if deadline is not None and time.time() > deadline:
            break
        f = basis @ c
        g = _grad_R2(c, basis, w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mh = m / (1 - beta1 ** t)
        vh = v / (1 - beta2 ** t)
        c = c - lr0 * mh / (np.sqrt(vh) + eps)
        R = _compute_R(basis @ c, w)
        if R < best_R:
            best_R = R
            best_c = c.copy()
    return best_c, best_R


def _newton_refine(c, basis, w, iters=30, deadline=None):
    """Newton / quasi-Newton refinement using numerical Hessian or damped steps."""
    c = c.copy()
    best_c = c.copy()
    best_R = _compute_R(basis @ c, w)
    n = c.size
    # Use simple damped gradient descent with adaptive step as refinement
    lr = 1e-3
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        f = basis @ c
        g = _grad_R2(c, basis, w)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        # Try step
        for _ in range(20):
            c_new = c - lr * g
            R_new = _compute_R(basis @ c_new, w)
            if R_new < best_R:
                c = c_new
                best_R = R_new
                best_c = c.copy()
                break
            lr *= 0.5
        else:
            break
        lr = min(lr * 1.5, 1e-2)
    return best_c, best_R


def _structured_starts():
    """Generate several structured initial coefficient vectors."""
    starts = []
    # all ones
    starts.append(np.ones(2 * _K + 1, dtype=np.float64))
    # only m=0
    c = np.zeros(2 * _K + 1); c[_K] = 1.0; starts.append(c)
    # only even m cos (m=0,2,4,...)
    c = np.zeros(2 * _K + 1)
    for m in range(0, _K + 1, 2):
        c[m + _K] = 1.0
    starts.append(c)
    # only odd m cos
    c = np.zeros(2 * _K + 1)
    for m in range(1, _K + 1, 2):
        c[m + _K] = 1.0
    starts.append(c)
    # symmetric cos only (m=0..10), no sine
    c = np.zeros(2 * _K + 1)
    for m in range(0, _K + 1):
        c[m + _K] = 1.0
    starts.append(c)
    # alternating signs
    c = np.ones(2 * _K + 1)
    for i in range(len(c)):
        if i % 2:
            c[i] = -1.0
    starts.append(c)
    # random few
    rng = np.random.default_rng(12345)
    for _ in range(4):
        starts.append(rng.standard_normal(2 * _K + 1))
    return starts


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + _DEADLINE - 20.0  # 20s safety margin
    basis, w = _build_basis()

    starts = _structured_starts()
    best_c = None
    best_R = np.inf

    # Phase 1: Adam from each start
    for c0 in starts:
        if time.time() > deadline:
            break
        c, R = _adam(c0, basis, w, lr0=0.05, iters=1500, deadline=deadline)
        if R < best_R:
            best_R = R
            best_c = c

    # Phase 2: refine best with Newton-like
    if best_c is not None:
        c, R = _newton_refine(best_c, basis, w, iters=40, deadline=deadline)
        if R < best_R:
            best_R = R
            best_c = c

    # Phase 3: a few more Adam from perturbed best
    if best_c is not None and time.time() < deadline:
        rng = np.random.default_rng(999)
        for _ in range(6):
            if time.time() > deadline:
                break
            c0 = best_c + 0.05 * rng.standard_normal(2 * _K + 1)
            c, R = _adam(c0, basis, w, lr0=0.02, iters=1000, deadline=deadline)
            if R < best_R:
                best_R = R
                best_c = c

    if best_c is None:
        best_c = np.ones(2 * _K + 1, dtype=np.float64)

    # Validate
    if best_c.shape != (2 * _K + 1,):
        best_c = np.ones(2 * _K + 1, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c / mx * 1.0
    if mx > 1e5:
        best_c = best_c / mx * 1e5

    return best_c.astype(np.float64)


# EVOLVE-BLOCK-END
