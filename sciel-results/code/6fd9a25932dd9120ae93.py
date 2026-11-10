# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_quadrature(n_theta=70, n_phi=130):
    u, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    U, P = np.meshgrid(u, phi, indexing='ij')
    W = np.outer(w, np.full(n_phi, dphi))
    return u, phi, W


def _assoc_legendre_P(k, m, x):
    try:
        from scipy.special import lpmv
        return lpmv(m, k, x)
    except Exception:
        Pmm = np.ones_like(x)
        if m > 0:
            somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
            fact = 1.0
            for i in range(1, m + 1):
                Pmm *= -fact * somx2
                fact += 2.0
        if k == m:
            return Pmm
        Pmp1 = x * (2 * m + 1) * Pmm
        if k == m + 1:
            return Pmp1
        for kk in range(m + 2, k + 1):
            Pkp1 = ((2 * kk - 1) * x * Pmp1 - (kk + m - 1) * Pmm) / (kk - m)
            Pmm = Pmp1
            Pmp1 = Pkp1
        return Pmp1


def _build_basis(k=10, u=None, phi=None):
    if u is None or phi is None:
        u, phi, _ = _build_quadrature()
    U, P = np.meshgrid(u, phi, indexing='ij')
    n_theta, n_phi = U.shape
    B = np.zeros((n_theta, n_phi, 2 * k + 1), dtype=np.float64)
    for m in range(0, k + 1):
        Nm = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                     float(np.math.factorial(k - m)) /
                     float(np.math.factorial(k + m)))
        Pkm = _assoc_legendre_P(k, m, u)
        Pkm2 = Pkm[:, None]
        if m == 0:
            B[:, :, k] = Nm * Pkm2
        else:
            B[:, :, k + m] = np.sqrt(2.0) * Nm * Pkm2 * np.cos(m * phi)[None, :]
            B[:, :, k - m] = np.sqrt(2.0) * Nm * Pkm2 * np.sin(m * phi)[None, :]
    return B


def _f_and_norms(B, W, c):
    f = np.einsum('ijm,m->ij', B, c)
    f2 = f * f
    l2 = np.sum(f2 * W)
    l4 = np.sum(f2 * f2 * W)
    return f, l2, l4


def _objective(B, W, c):
    # minimize ||f||4 / ||f||2 ; return (R, f)
    f, l2, l4 = _f_and_norms(B, W, c)
    if l2 <= 0:
        return np.inf, f
    return np.sqrt(l4 / l2), f


def _grad_R2(B, W, c):
    f, l2, l4 = _f_and_norms(B, W, c)
    if l2 <= 0:
        return np.inf, np.zeros_like(c)
    R2 = l4 / l2
    # dR2/dc_j = (2/l2^2) * sum W f Y_j (2 f^2 l2 - l4)
    term = W * f * (2.0 * f * f * l2 - l4)
    grad = (2.0 / (l2 * l2)) * np.einsum('ij,ijm->m', term, B)
    return R2, grad


def _line_search(B, W, c, g, R2, max_iter=50):
    step = 1.0
    best_c = c.copy()
    best_R2 = R2
    for _ in range(max_iter):
        cand = c - step * g
        R2c, _ = _objective(B, W, cand)
        if np.isfinite(R2c) and R2c < best_R2:
            return cand, R2c
        step *= 0.5
        if step < 1e-14:
            break
    return best_c, best_R2


def rudin_problem10() -> np.ndarray:
    k = 10
    t0 = time.time()
    deadline = t0 + 570.0  # safety margin under 600s

    u, phi, W = _build_quadrature(n_theta=70, n_phi=130)
    B = _build_basis(k=k, u=u, phi=phi)

    best_c = None
    best_R = np.inf
    rng = np.random.default_rng(7)

    inits = [np.ones(2 * k + 1, dtype=np.float64)]
    z = np.zeros(2 * k + 1); z[k] = 1.0
    inits.append(z)
    for _ in range(10):
        inits.append(rng.normal(size=2 * k + 1).astype(np.float64))
    sym = np.zeros(2 * k + 1); sym[k:] = 1.0
    inits.append(sym)
    anti = np.zeros(2 * k + 1); anti[:k] = 1.0
    inits.append(anti)

    for c0 in inits:
        if time.time() > deadline:
            break
        c = c0.copy()
        R, _ = _objective(B, W, c)
        if np.isfinite(R) and (best_c is None or R < best_R):
            best_R = R
            best_c = c.copy()

        R2, g = _grad_R2(B, W, c)
        it = 0
        while time.time() < deadline and it < 5000:
            it += 1
            gn = np.linalg.norm(g)
            if not np.isfinite(gn) or gn < 1e-13:
                break
            c, R2 = _line_search(B, W, c, g, R2)
            R, _ = _objective(B, W, c)
            if np.isfinite(R) and R < best_R:
                best_R = R
                best_c = c.copy()
            R2, g = _grad_R2(B, W, c)

    if best_c is None:
        best_c = np.ones(2 * k + 1, dtype=np.float64)

    # Polish
    c = best_c.copy()
    for _ in range(300):
        if time.time() > deadline:
            break
        R2, g = _grad_R2(B, W, c)
        gn = np.linalg.norm(g)
        if not np.isfinite(gn) or gn < 1e-15:
            break
        c, R2 = _line_search(B, W, c, g, R2)
        R, _ = _objective(B, W, c)
        if np.isfinite(R) and R < best_R:
            best_R = R
            best_c = c.copy()

    mc = best_c.copy()
    mx = np.max(np.abs(mc))
    if not np.isfinite(mx) or mx <= 0:
        mc = np.ones(2 * k + 1, dtype=np.float64)
        mx = 1.0
    if mx < 1e-3:
        mc = mc * (1e-3 / mx)
    elif mx > 1e5:
        mc = mc * (1e5 / mx)

    return mc.astype(np.float64)


# EVOLVE-BLOCK-END
