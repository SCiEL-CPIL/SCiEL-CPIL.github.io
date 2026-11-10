# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial


def _build_harmonics():
    k = 10
    m_vals = np.arange(-k, k + 1)
    # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    N = np.sqrt(21.0 / (4.0 * np.pi) *
                factorial(k - np.abs(m_vals)) / factorial(k + np.abs(m_vals)))
    return m_vals, N


def _eval_coeffs(c, mu, phi, m_vals, N):
    """Evaluate f = sum c_m Y_m on the grid (mu=cos theta, phi)."""
    f = np.zeros_like(mu)
    for i, m in enumerate(m_vals):
        cm = c[i]
        if cm == 0.0:
            continue
        am = abs(m)
        P = lpmv(am, k, mu) * N[i]
        if m == 0:
            f += cm * P
        elif m > 0:
            f += cm * np.sqrt(2.0) * P * np.cos(m * phi)
        else:
            f += cm * np.sqrt(2.0) * P * np.sin(am * phi)
    return f


def _project_to_coeffs(f, mu, phi, w, m_vals, N):
    """Least-squares projection of grid values f back to degree-10 coeffs."""
    k = 10
    M = len(m_vals)
    A = np.empty((len(mu), M))
    for i, m in enumerate(m_vals):
        am = abs(m)
        P = lpmv(am, k, mu) * N[i]
        if m == 0:
            A[:, i] = P
        elif m > 0:
            A[:, i] = np.sqrt(2.0) * P * np.cos(m * phi)
        else:
            A[:, i] = np.sqrt(2.0) * P * np.sin(am * phi)
    W = w
    AWT = (A * W[:, None]).T
    c = np.linalg.lstsq(AWT, f * W, rcond=None)[0]
    return c


def _ratio(c, mu, phi, w, m_vals, N):
    f = _eval_coeffs(c, mu, phi, m_vals, N)
    l2 = np.sqrt(np.sum(w * f * f))
    l4 = np.sum(w * f ** 4) ** 0.25
    return l4 / l2


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    DEADLINE = 600.0
    k = 10

    # Quadrature grid: 70-point Gauss-Legendre in cos(theta) x 130 phi
    n_mu, n_phi = 70, 130
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    # map [-1,1] to cos theta in [-1,1] (already), weight w for dmu
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    W = (w[:, None] * dphi)

    m_vals, N = _build_harmonics()

    # Start from all-ones
    c = np.ones(2 * k + 1, dtype=np.float64)

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---- Power iteration: c <- project(|f|^3), with over-relaxation ----
    best_c = c.copy()
    best_r = _ratio(c, MU, PHI, W, m_vals, N)

    relax = 1.0
    it = 0
    while time_left() > 30.0:
        f = _eval_coeffs(c, MU, PHI, m_vals, N)
        g = np.abs(f) ** 3.0
        c_new = _project_to_coeffs(g, MU, PHI, W, m_vals, N)
        c_try = (1.0 - relax) * c + relax * c_new
        r_try = _ratio(c_try, MU, PHI, W, m_vals, N)
        if r_try < best_r:
            best_r = r_try
            best_c = c_try.copy()
        c = c_try
        it += 1
        # adapt relaxation
        if it % 40 == 0:
            relax = max(0.3, relax * 0.97)

    # ---- Gradient polish: minimize R = l4/l2 via analytic gradient ----
    def ratio_and_grad(c):
        f = _eval_coeffs(c, MU, PHI, m_vals, N)
        l2sq = np.sum(W * f * f)
        l2 = np.sqrt(l2sq)
        l4sq2 = np.sum(W * f ** 4)
        l4 = l4sq2 ** 0.25
        R = l4 / l2
        # dR/dc_i = (1/l2) d(l4)/dc_i - (l4/l2^2) d(l2)/dc_i
        # d(l4)/dc_i = (1/(4 l4^3)) * 4 * sum W f^3 Y_i = (1/l4^3) sum W f^3 Y_i
        # d(l2)/dc_i = (1/l2) sum W f Y_i
        # => dR/dc_i = (1/(l2 l4^3)) sum W f^3 Y_i - (l4/(l2^3)) sum W f Y_i
        # Build Y_i on grid
        grad = np.zeros(2 * k + 1)
        for i, m in enumerate(m_vals):
            am = abs(m)
            P = lpmv(am, k, MU) * N[i]
            if m == 0:
                Yi = P
            elif m > 0:
                Yi = np.sqrt(2.0) * P * np.cos(m * PHI)
            else:
                Yi = np.sqrt(2.0) * P * np.sin(am * PHI)
            s1 = np.sum(W * f ** 3 * Yi)
            s2 = np.sum(W * f * Yi)
            grad[i] = (1.0 / (l2 * l4 ** 3)) * s1 - (l4 / l2 ** 3) * s2
        return R, grad

    c = best_c.copy()
    lr = 1e-3
    stall = 0
    while time_left() > 15.0:
        R, grad = ratio_and_grad(c)
        if R < best_r:
            best_r = R
            best_c = c.copy()
            stall = 0
        else:
            stall += 1
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        step = lr * grad / gnorm
        c = c - step
        if stall > 30:
            lr *= 0.5
            stall = 0
            if lr < 1e-9:
                break

    c = best_c
    # Final validation of bounds
    if not (np.isfinite(c).all()):
        c = np.ones(2 * k + 1)
    mx = np.max(np.abs(c))
    if mx < 1e-3:
        c = c / mx if mx > 0 else np.ones(2 * k + 1)
        c = c / np.max(np.abs(c))
    if np.max(np.abs(c)) > 1e5:
        c = c * (1e4 / np.max(np.abs(c)))
    return c


# EVOLVE-BLOCK-END
