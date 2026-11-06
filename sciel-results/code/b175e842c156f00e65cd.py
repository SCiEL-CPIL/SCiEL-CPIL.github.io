"""Cosmology growth-index fit: structural redesign with numerical ODE reference + bounded Nelder-Mead."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# ---------- cosmology helpers ----------
def _H2(x, Om0, w0, wa):
    """H^2 in units of H0^2 as a function of x=ln(a)."""
    e = np.exp(x)
    return Om0 * e ** (-3.0 * (1.0 + w0 + wa)) + (1.0 - Om0) * e ** (-3.0 * (1.0 + w0))

def _Om(x, Om0, w0, wa):
    return Om0 * np.exp(-3.0 * (1.0 + w0 + wa) * x) / _H2(x, Om0, w0, wa)

def _rhs(x, y, Om0, w0, wa):
    """Second-order ODE for D(a): d2D/dx2 = (3/2)Om - (1/2)q_eff * dD/dx."""
    h2 = _H2(x, Om0, w0, wa)
    om = _Om(x, Om0, w0, wa)
    # q_eff = -0.5 * (d ln H2 / d ln a) = -0.5 * (d ln H2 / dx)
    # d ln H2 / dx = (-3(1+w0+wa)Om0 e^{-3(1+w0+wa)x} + -3(1+w0)(1-Om0) e^{-3(1+w0)x}) / H2
    a = np.exp(x)
    dlnH2 = (-3.0 * (1.0 + w0 + wa) * Om0 * np.exp(-3.0 * (1.0 + w0 + wa) * x)
             - 3.0 * (1.0 + w0) * (1.0 - Om0) * np.exp(-3.0 * (1.0 + w0) * x)) / h2
    qeff = -0.5 * dlnH2
    dD = y[1]
    return np.array([dD, 1.5 * om - 0.5 * qeff * dD])

def _solve_D(xgrid, Om0, w0, wa):
    """Integrate ODE from x=0 backward to xgrid (which is <= 0). Return D normalized D(1)=1."""
    # Integrate from x=0 to x_min using RK4 on the 2nd-order system.
    # State y = [D, dD/dx]. At x=0: D=1. Choose dD/dx(0) via the asymptotic growth rate.
    # f(0)=Om0^gamma where gamma ~ 0.55; dD/dx = f*D at x=0.
    gamma0 = 0.55
    f0 = Om0 ** gamma0
    y = np.array([1.0, f0])
    x0 = 0.0
    xmin = float(xgrid.min())
    # step size
    n = 2000
    hstep = (xmin - x0) / n
    # integrate backward
    D = np.empty_like(xgrid)
    # we'll integrate from x0 down to xmin, storing at xgrid points
    # xgrid is sorted ascending (most negative first). We integrate in decreasing x.
    # Build list of target points in order of integration (x0 down to xmin).
    order = np.argsort(xgrid)[::-1]  # descending x (0 first)
    targets = xgrid[order]
    idx = 0
    cur_x = x0
    cur_y = y.copy()
    results = np.empty(len(xgrid))
    for ti, tx in enumerate(targets):
        while cur_x > tx + 1e-15:
            # one RK4 step from cur_x to cur_x - |hstep| but not below tx
            step = min(-hstep, tx - cur_x)
            if step < -1e-15:
                k1 = _rhs(cur_x, cur_y, Om0, w0, wa)
                k2 = _rhs(cur_x + 0.5 * step, cur_y + 0.5 * step * k1, Om0, w0, wa)
                k3 = _rhs(cur_x + 0.5 * step, cur_y + 0.5 * step * k2, Om0, w0, wa)
                k4 = _rhs(cur_x + step, cur_y + step * k3, Om0, w0, wa)
                cur_y = cur_y + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                cur_x += step
        results[order[ti]] = cur_y[0]
    return results

def _gamma_model(c, a, Om, w0, wa):
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q

def _f_model(c, a, Om, w0, wa):
    g = _gamma_model(c, a, Om, w0, wa)
    return Om ** g

def _D_model(c, xgrid, Om0, w0, wa):
    """D(a)/D(1) via exp(-int f dx) from x to 0."""
    a = np.exp(xgrid)
    Om = _Om(xgrid, Om0, w0, wa)
    f = _f_model(c, a, Om, w0, wa)
    # integrate f from x to 0: D/D1 = exp(-int_x^0 f dx)
    # cumulative integral of f over xgrid (ascending). 
    # We want int from x_i to 0 = (int from x_min to 0) - (int from x_min to x_i)
    # Use cumulative trapezoid.
    # xgrid ascending, f corresponding.
    dx = np.diff(xgrid)
    trap = 0.5 * (f[:-1] + f[1:]) * dx
    cum = np.concatenate([[0.0], np.cumsum(trap)])
    total = cum[-1]
    int_x_to_0 = total - cum
    return np.exp(-int_x_to_0)

def _score(c, cosmologies, xgrid):
    """Return worst relative error * 100 over f and D."""
    worst = 0.0
    for (Om0, w0, wa) in cosmologies:
        a = np.exp(xgrid)
        Om = _Om(xgrid, Om0, w0, wa)
        # f reference: from exact ODE dD/dx / D
        # We have D from ODE; dD/dx from ODE state. Recompute dD/dx.
        D_ref = _solve_D(xgrid, Om0, w0, wa)
        # dD/dx at each point: recompute via ODE state; approximate by derivative
        # Simpler: f_ref = d ln D / dx. Compute numerically from D_ref.
        # Use central differences on xgrid.
        dDdx = np.gradient(D_ref, xgrid)
        f_ref = dDdx / D_ref
        f_mod = _f_model(c, a, Om, w0, wa)
        # guard against non-finite
        if not np.all(np.isfinite(f_mod)) or not np.all(np.isfinite(f_ref)):
            return 1e6
        rel_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-12)
        worst = max(worst, float(np.max(rel_f)))
        # D reference normalized
        D_ref_norm = D_ref / D_ref[-1]
        D_mod = _D_model(c, xgrid, Om0, w0, wa)
        if not np.all(np.isfinite(D_mod)):
            return 1e6
        rel_D = np.abs(D_mod - D_ref_norm) / np.maximum(np.abs(D_ref_norm), 1e-12)
        worst = max(worst, float(np.max(rel_D)))
    return 100.0 * worst

def _nelder_mead(func, x0, bounds, max_iter=2000, tol=1e-10, deadline=None):
    """Bounded Nelder-Mead."""
    n = len(x0)
    # build initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        step = 0.1 * (bounds[1][i] - bounds[0][i]) if (bounds[1][i] - bounds[0][i]) > 0 else 0.1
        p[i] += step
        simplex.append(p)
    vals = [func(p) for p in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        # centroid of all but worst
        cen = np.mean(simplex[:-1], axis=0)
        # reflect
        xr = cen + alpha * (cen - simplex[-1])
        fxr = func(xr)
        if fxr < vals[0]:
            # expand
            xe = cen + gamma * (cen - simplex[-1])
            fxe = func(xe)
            if fxe < fxr:
                simplex[-1], vals[-1] = xe, fxe
            else:
                simplex[-1], vals[-1] = xr, fxr
        elif fxr < vals[-2]:
            simplex[-1], vals[-1] = xr, fxr
        else:
            # contract
            xc = cen + rho * (simplex[-1] - cen)
            fxc = func(xc)
            if fxc < vals[-1]:
                simplex[-1], vals[-1] = xc, fxc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = func(simplex[i])
        it += 1
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]]

def construct():
    # deadline guard
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s

    # Validation cosmologies (plausible CPL set)
    cosmologies = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -1.0, 0.3),   # CPL
        (0.25, -0.9, 0.2),
        (0.35, -0.95, 0.1),
        (0.3, -0.8, 0.4),
        (0.28, -1.0, 0.0),
    ]
    # xgrid: a from 0.1 to 1.0
    a_grid = np.linspace(0.1, 1.0, 40)
    xgrid = np.log(a_grid)

    bounds = (np.array([-10.0] * 6), np.array([10.0] * 6))
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Precompute references to speed up scoring
    # (already computed on the fly; keep simple)

    best_x = x0
    best_val = _score(x0, cosmologies, xgrid)

    # Multi-start Nelder-Mead with a few perturbations
    starts = [x0]
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = x0 + rng.normal(0, 0.5, size=6)
        p = np.clip(p, bounds[0], bounds[1])
        starts.append(p)

    for s in starts:
        if time.time() > deadline:
            break
        x, v = _nelder_mead(_score, s, bounds, max_iter=800,
                            tol=1e-9, deadline=deadline)
        if v < best_val:
            best_val = v
            best_x = x

    # Enforce gamma in [0.1, 1.2] at a sample of locations by clamping coefficients softly.
    # Check gamma range on a fine grid for the worst cosmology; if violated, scale down.
    a_fine = np.linspace(0.05, 1.0, 200)
    for (Om0, w0, wa) in cosmologies:
        x = np.log(a_fine)
        Om = _Om(x, Om0, w0, wa)
        g = _gamma_model(best_x, a_fine, Om, w0, wa)
        if g.min() < 0.1 or g.max() > 1.2:
            # shrink coefficients toward constant 0.55
            scale = 0.5
            best_x = 0.55 + scale * (best_x - 0.55)

    # Final clip
    best_x = np.clip(best_x, bounds[0], bounds[1])
    return [float(v) for v in best_x]
# EVOLVE-BLOCK-END
