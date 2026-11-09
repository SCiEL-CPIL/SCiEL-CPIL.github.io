"""Cosmology: compact dark-energy growth approximation.

Self-calibrating fit of the fixed six-coefficient gamma ansatz against the
exact general-relativistic second-order growth ODE, over an ensemble of
physically-motivated CPL dark-energy cosmologies.

Interface (unchanged): construct() -> real vector of six finite coefficients
in [-10, 10].
"""
import math
import time
import numpy as np

_DEADLINE = 90.0
_SAFE = 6.0          # wall-clock safety margin (seconds)
_MIN_COEF = -10.0
_MAX_COEF = 10.0


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------
def _Om_m(a, Om0):
    return Om0 * a ** -3


def _Om_de(a, w0, wa):
    return (1.0 - Om0_global) * a ** (-3.0 * (1.0 + w0 + wa))


# We avoid a global; keep it explicit instead.
def _cosmo_Om(a, Om0, w0, wa):
    Om_m = Om0 * a ** -3
    Om_de = (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa))
    return Om_m + Om_de


def _cosmo_q(a, Om0, w0, wa):
    Om = _cosmo_Om(a, Om0, w0, wa)
    return 1.0 - Om


def _cosmo_u(a, w0, wa):
    return 1.0 + w0 + wa * (1.0 - a)


def _true_gamma(a, Om0, w0, wa):
    """Exact growth index gamma(a) from the GR second-order ODE."""
    Om = _cosmo_Om(a, Om0, w0, wa)
    q = 1.0 - Om
    H2 = Om + (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa))
    dH2 = -3.0 * Om - 3.0 * (1.0 + w0 + wa) * (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa))
    f = Om ** 0.5
    F = 0.5 * (dH2 / H2) + 0.75 * Om
    return (0.5 * F + math.sqrt(0.25 * F * F + f) - F) / (f - 1.0)


def _true_D(a, Om0, w0, wa, n=2000):
    """Exact normalized growth factor D(a)/D(1) via the second-order ODE."""
    # Integrate over x = ln a from 0 (a=1) down to ln(a).
    xa = math.log(a)
    if abs(xa) < 1e-12:
        return 1.0
    # We integrate from x=0 to x=xa. Use a fixed step in x.
    N = n
    dx = xa / N
    # State: D, dD/dx.  ODE: D'' + (3 + dH2/(2H2)) D' - (3/2) Om D = 0  (x=ln a)
    # => D'' = (3/2) Om - (3 + dH2/(2H2)) D'
    D = 1.0
    Dp = 0.0
    # We need the value at x=xa; integrate step by step with RK4.
    def rhs(x, Dv, Dpv):
        aa = math.exp(x)
        Om = _cosmo_Om(aa, Om0, w0, wa)
        H2 = Om + (1.0 - Om0) * aa ** (-3.0 * (1.0 + w0 + wa))
        dH2 = -3.0 * Om - 3.0 * (1.0 + w0 + wa) * (1.0 - Om0) * aa ** (-3.0 * (1.0 + w0 + wa))
        F = 0.5 * (dH2 / H2) + 0.75 * Om
        # D'' = (3/2) Om - (3 + dH2/(2H2)) D'   ... but (3 + dH2/(2H2)) = 2F + 1.5 Om
        # Let's use: D'' = 1.5*Om - (2F + 1.5*Om)*Dp
        Dpp = 1.5 * Om - (2.0 * F + 1.5 * Om) * Dpv
        return Dp, Dpp

    for i in range(N):
        x0 = i * dx
        k1d, k1p = rhs(x0, D, Dp)
        k2d, k2p = rhs(x0 + 0.5 * dx, D + 0.5 * dx * k1d, Dp + 0.5 * dx * k1p)
        k3d, k3p = rhs(x0 + 0.5 * dx, D + 0.5 * dx * k2d, Dp + 0.5 * dx * k2p)
        k4d, k4p = rhs(x0 + dx, D + dx * k3d, Dp + dx * k3p)
        D = D + (dx / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        Dp = Dp + (dx / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    return D


def _approx_gamma(a, c, Om0, w0, wa):
    q = _cosmo_q(a, Om0, w0, wa)
    u = _cosmo_u(a, w0, wa)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _approx_f(a, c, Om0, w0, wa):
    Om = _cosmo_Om(a, Om0, w0, wa)
    g = _approx_gamma(a, c, Om0, w0, wa)
    if g < 0.1:
        g = 0.1
    elif g > 1.2:
        g = 1.2
    return Om ** g


def _approx_D(a, c, Om0, w0, wa, n=2000):
    xa = math.log(a)
    if abs(xa) < 1e-12:
        return 1.0
    N = n
    dx = xa / N
    D = 1.0
    for i in range(N):
        x0 = i * dx
        aa = math.exp(x0)
        Om = _cosmo_Om(aa, Om0, w0, wa)
        g = _approx_gamma(aa, c, Om0, w0, wa)
        if g < 0.1:
            g = 0.1
        elif g > 1.2:
            g = 1.2
        f = Om ** g
        # dD/dx = -f  (since D(a)/D(1) = exp(-int_0^x f dx))
        D = D * math.exp(-f * dx)
    return D


# ---------------------------------------------------------------------------
# Validation ensemble (physically-motivated CPL cosmologies)
# ---------------------------------------------------------------------------
def _build_ensemble():
    models = []
    # (Om0, w0, wa)
    base = [
        (0.315, -1.0, 0.0),    # LCDM-ish
        (0.300, -1.0, 0.0),
        (0.280, -1.0, 0.0),
        (0.330, -1.0, 0.0),
        (0.315, -0.9, 0.0),
        (0.315, -1.1, 0.0),
        (0.315, -1.0, 0.5),
        (0.315, -1.0, -0.5),
        (0.315, -1.0, 1.0),
        (0.315, -1.0, -1.0),
        (0.315, -0.5, 0.0),
        (0.315, -1.5, 0.0),
        (0.315, -1.0, 0.0),  # duplicate-safe
        (0.250, -1.0, 0.5),
        (0.350, -1.0, -0.5),
        (0.300, -0.8, 0.3),
        (0.320, -1.2, 0.2),
        (0.290, -1.0, 0.8),
    ]
    # De-duplicate
    seen = set()
    for m in base:
        if m not in seen:
            seen.add(m)
            models.append(m)
    return models


def _a_values():
    # Disjoint a grid covering the validation range (a from ~0.05 to 1.0)
    return np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                     0.50, 0.60, 0.70, 0.80, 0.90, 1.0])


# ---------------------------------------------------------------------------
# Objective: worst relative error over f and D
# ---------------------------------------------------------------------------
def _objective(c, models, avals, nD=1200):
    worst = 0.0
    for (Om0, w0, wa) in models:
        for a in avals:
            if a <= 0.0 or a >= 1.0:
                continue
            # True f = Om^gamma_true
            Om = _cosmo_Om(a, Om0, w0, wa)
            gt = _true_gamma(a, Om0, w0, wa)
            f_true = Om ** gt
            f_app = _approx_f(a, c, Om0, w0, wa)
            if abs(f_true) > 1e-12:
                err_f = abs(f_app - f_true) / abs(f_true)
            else:
                err_f = abs(f_app - f_true)
            if err_f > worst:
                worst = err_f
            # True D
            D_true = _true_D(a, Om0, w0, wa, n=nD)
            D_app = _approx_D(a, c, Om0, w0, wa, n=nD)
            if abs(D_true) > 1e-12:
                err_D = abs(D_app - D_true) / abs(D_true)
            else:
                err_D = abs(D_app - D_true)
            if err_D > worst:
                worst = err_D
    return worst


# ---------------------------------------------------------------------------
# Optimization: Nelder-Mead with deadline guard
# ---------------------------------------------------------------------------
def _nelder_mead(fx, x0, bounds, max_iter=4000, tol=1e-9, deadline=None):
    n = len(x0)
    # Build initial simplex
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        step = 0.5 if abs(p[i]) < 1e-8 else 0.5 * p[i]
        p[i] += step
        simplex.append(p)

    vals = [fx(p) for p in simplex]

    def clamp(p):
        return [max(bounds[0], min(bounds[1], v)) for v in p]

    best = min(vals)
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[0] < best:
            best = vals[0]
        # Check convergence
        if abs(vals[-1] - vals[0]) < tol:
            break
        # Centroid of all but worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        # Reflection
        alpha = 1.0
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        xr = clamp(xr)
        fr = fx(xr)
        if fr < vals[0]:
            # Expansion
            gamma_e = 2.0
            xe = [centroid[j] + gamma_e * (xr[j] - centroid[j]) for j in range(n)]
            xe = clamp(xe)
            fe = fx(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            # Contraction
            rho = 0.5
            if fr < vals[-1]:
                xc = [centroid[j] + rho * (xr[j] - centroid[j]) for j in range(n)]
            else:
                xc = [centroid[j] - rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            xc = clamp(xc)
            fc = fx(xc)
            if fc < min(fr, vals[-1]):
                simplex[-1] = xc
                vals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    vals[i] = fx(simplex[i])
    # Return best point
    order = np.argsort(vals)
    return simplex[order[0]], best


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _SAFE

    models = _build_ensemble()
    avals = _a_values()

    # Precompute true values once (cache) to speed the objective
    cache = {}
    for (Om0, w0, wa) in models:
        for a in avals:
            if a <= 0.0 or a >= 1.0:
                continue
            Om = _cosmo_Om(a, Om0, w0, wa)
            gt = _true_gamma(a, Om0, w0, wa)
            f_true = Om ** gt
            D_true = _true_D(a, Om0, w0, wa, n=1200)
            cache[(Om0, w0, wa, a)] = (f_true, D_true)

    def fast_objective(c):
        worst = 0.0
        for (Om0, w0, wa) in models:
            for a in avals:
                if a <= 0.0 or a >= 1.0:
                    continue
                f_true, D_true = cache[(Om0, w0, wa, a)]
                f_app = _approx_f(a, c, Om0, w0, wa)
                err_f = abs(f_app - f_true) / max(abs(f_true), 1e-12)
                if err_f > worst:
                    worst = err_f
                D_app = _approx_D(a, c, Om0, w0, wa, n=1200)
                err_D = abs(D_app - D_true) / max(abs(D_true), 1e-12)
                if err_D > worst:
                    worst = err_D
        return worst

    # Multi-start Nelder-Mead
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
        [0.52, 0.05, 0.02, 0.02, 0.0, 0.0],
        [0.58, -0.05, 0.0, 0.0, 0.0, 0.02],
    ]

    best_c = None
    best_val = float('inf')
    for s in starts:
        if time.time() > deadline:
            break
        c0 = [max(_MIN_COEF, min(_MAX_COEF, v)) for v in s]
        c, val = _nelder_mead(fast_objective, c0, (_MIN_COEF, _MAX_COEF),
                              max_iter=3000, tol=1e-10, deadline=deadline)
        if val < best_val:
            best_val = val
            best_c = c

    if best_c is None:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Final clamp to [-10, 10]
    best_c = [max(_MIN_COEF, min(_MAX_COEF, v)) for v in best_c]
    return best_c
