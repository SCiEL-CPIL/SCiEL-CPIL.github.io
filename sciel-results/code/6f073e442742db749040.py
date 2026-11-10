# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

# ---- constants ----
K = 10
NTHETA = 70
NPHI = 130
DEADLINE = 600.0
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _precompute_basis():
    """Return (Y, w) where Y has shape (NPHI, NTHETA, 21) and w shape (NPHI, NTHETA)."""
    # theta in [0, pi], phi in [0, 2pi)
    theta = np.linspace(0.0, np.pi, NTHETA)
    phi = np.linspace(0.0, 2.0 * np.pi, NPHI, endpoint=False)
    ct = np.cos(theta)
    st = np.sin(theta)

    # Condon-Shortley phase (-1)^m
    phase = np.array([(-1.0) ** m for m in range(K + 1)])

    # normalized associated Legendre P_l^m(x), with CS phase
    P = np.zeros((K + 1, NTHETA))
    for m in range(K + 1):
        P[m, :] = phase[m] * lpmv(m, K, ct)

    # normalization N_m
    N = np.zeros(K + 1)
    for m in range(K + 1):
        N[m] = np.sqrt((2 * K + 1) / (4.0 * np.pi)
                       * float(np.math.factorial(K - m)) / float(np.math.factorial(K + m)))

    # weights: dOmega = st dtheta dphi, theta uniform so dtheta = pi/NTHETA
    dtheta = np.pi / NTHETA
    dphi = 2.0 * np.pi / NPHI
    w_theta = st * dtheta
    w = np.outer(dphi, w_theta)  # (NPHI, NTHETA)

    # Build basis: index j = m + K for m = -K..K
    Y = np.zeros((NPHI, NTHETA, 2 * K + 1))
    # m = 0
    Y[:, :, K] = N[0] * P[0, :]
    for m in range(1, K + 1):
        base = N[m] * P[m, :]
        Y[:, :, K + m] = np.sqrt(2.0) * base * np.cos(m * phi)[:, None]
        Y[:, :, K - m] = np.sqrt(2.0) * base * np.sin(m * phi)[:, None]
    return Y, w


def _objective_and_grad(c, Y, w, wsum):
    """J = 0.5 * (L4^2 / L2^2) and its gradient w.r.t. c."""
    f = np.tensordot(Y, c, axes=([2], [0]))  # (NPHI, NTHETA)
    L2 = np.sum(w * f * f)
    L4 = np.sum(w * f ** 4)
    if L2 <= 1e-300:
        return 1e300, np.zeros_like(c)
    J = 0.5 * (L4 / L2)
    # dJ/dc_i = (1/L2) * sum w ( f^3 Y_i ) - (L4/L2^2) * sum w ( f Y_i )
    s1 = np.tensordot(w * f ** 3, Y, axes=([0, 1], [0, 1]))  # (21,)
    s2 = np.tensordot(w * f, Y, axes=([0, 1], [0, 1]))        # (21,)
    grad = s1 / L2 - (L4 / (L2 * L2)) * s2
    return J, grad


def _objective(c, Y, w, wsum):
    J, _ = _objective_and_grad(c, Y, w, wsum)
    return J


def _compute_R(c, Y, w):
    f = np.tensordot(Y, c, axes=([2], [0]))
    L2 = np.sum(w * f * f)
    L4 = np.sum(w * f ** 4)
    if L2 <= 1e-300:
        return 1e300
    return np.sqrt(L4 / L2)


def _make_starts():
    n = 2 * K + 1
    idx = np.arange(-K, K + 1)
    starts = []
    # all ones (seed baseline)
    starts.append(np.ones(n, dtype=np.float64))
    # cos pattern
    starts.append(np.cos(idx * 0.3))
    starts.append(np.cos(idx * 0.7))
    starts.append(np.cos(idx * 1.1))
    # sin pattern
    starts.append(np.sin(idx * 0.5))
    starts.append(np.sin(idx * 0.9))
    # alternating
    starts.append(np.array([(-1.0) ** i for i in range(n)]))
    starts.append(np.array([1.0 if (i % 2 == 0) else -0.5 for i in range(n)]))
    # sparse: only even m
    s = np.zeros(n); s[::2] = 1.0; starts.append(s)
    # sparse: only odd m
    s = np.zeros(n); s[1::2] = 1.0; starts.append(s)
    # only |m| <= 2
    s = np.zeros(n); s[(idx >= -2) & (idx <= 2)] = 1.0; starts.append(s)
    # only high |m|
    s = np.zeros(n); s[np.abs(idx) >= 8] = 1.0; starts.append(s)
    # random
    rng = np.random.default_rng(12345)
    for _ in range(8):
        starts.append(rng.standard_normal(n))
    # normalize each to unit L2-ish (scale by 1/max|c| to keep in [1e-3,1e5])
    out = []
    for s in starts:
        mx = np.max(np.abs(s))
        if mx < 1e-3:
            s = s * (1.0 / mx) if mx > 0 else s + 1.0
        elif mx > 1.0:
            s = s * (1.0 / mx)
        out.append(s.astype(np.float64))
    return out


def rudin_problem10() -> np.ndarray:
    Y, w = _precompute_basis()
    wsum = float(np.sum(w))

    starts = _make_starts()

    best_c = None
    best_R = 1e300

    def consider(c):
        nonlocal best_c, best_R
        if c is None:
            return
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (2 * K + 1,):
            return
        if not np.all(np.isfinite(c)):
            return
        mx = np.max(np.abs(c))
        if mx < 1e-3 or mx > 1e5:
            return
        R = _compute_R(c, Y, w)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Phase 1: multi-start L-BFGS-B with moderate tolerance
    for s in starts:
        if _time_left() < 15.0:
            break
        try:
            res = minimize(
                _objective, s, args=(Y, w, wsum),
                jac=_objective_and_grad,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 30, 'maxfun': 4000},
            )
            consider(res.x)
        except Exception:
            pass
        consider(s)  # keep the start too

    # Phase 2: high-precision polish on the best point
    if best_c is not None and _time_left() > 20.0:
        try:
            res = minimize(
                _objective, best_c, args=(Y, w, wsum),
                jac=_objective_and_grad,
                method='L-BFGS-B',
                options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-10,
                         'maxls': 50, 'maxfun': 50000},
            )
            consider(res.x)
        except Exception:
            pass

    # Safety: ensure validity
    if best_c is None:
        best_c = np.ones(2 * K + 1, dtype=np.float64)
    else:
        best_c = np.asarray(best_c, dtype=np.float64)
        if not np.all(np.isfinite(best_c)):
            best_c = np.ones(2 * K + 1, dtype=np.float64)
        mx = np.max(np.abs(best_c))
        if mx < 1e-3:
            best_c = best_c * (1.0 / mx) if mx > 0 else np.ones(2 * K + 1)
        elif mx > 1e5:
            best_c = best_c * (1.0 / mx)

    return best_c


# EVOLVE-BLOCK-END
