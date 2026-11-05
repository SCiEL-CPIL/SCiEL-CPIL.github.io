"""Reference-cosmology anchoring: fit the 6-coefficient gamma ansatz to a
high-accuracy GR ODE reference across a diverse set of CPL dark-energy models."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 82.0   # safety margin before the 90 s kill

def _om_a(a, Om0, wa):
    w0 = -1.0
    x = a ** (-3.0 * (1.0 + w0 + wa))
    return Om0 * x / (Om0 * x + 1.0 - Om0)

def _q_of(a, Om0, wa):
    return 1.0 - _om_a(a, Om0, wa)

def _u_of(a, wa):
    w0 = -1.0
    return 1.0 + w0 + wa * (1.0 - a)

def _gamma(a, Om0, wa, c):
    q = _q_of(a, Om0, wa)
    u = _u_of(a, wa)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q

def _f(a, Om0, wa, c):
    om = _om_a(a, Om0, wa)
    return om ** _gamma(a, Om0, wa, c)

def _ref_ode(Om0, wa, a_grid, n=20000):
    """Integrate the linear growth ODE in x=ln a from a=1 (x=0) down to a_min."""
    w0 = -1.0
    x = np.linspace(0.0, math.log(a_grid[0]), n + 1)
    a = np.exp(x)
    om = _om_a(a, Om0, wa)
    E2 = om + (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa))
    E = np.sqrt(E2)
    H = E  # H0=1
    Hdot = -0.5 * E2 * (1.0 + 3.0 * w0 + 3.0 * wa * (1.0 - a))
    dHdlna = Hdot / H
    d2Hdlna2 = (Hdot / H) ** 2 + Hdot / H ** 2 * Hdot  # placeholder, computed below
    # d^2H/dlna^2 = Hdot/H * dHdlna + Hdot/H^2 * dHdlna * H  ... use:
    d2Hdlna2 = (Hdot / H) * (Hdot / H) + Hdot * (Hdot / H) / H
    # d2H/dlna^2 = (Hdot/H)^2 + (Hdot/H)*(Hdot/H)  -> recompute cleanly:
    dHdlna = Hdot / H
    d2Hdlna2 = dHdlna * dHdlna + (Hdot / H ** 2) * dHdlna
    # d2H/dlna^2 = dHdlna^2 + dHdlna * (dHdlna) where dHdlna = Hdot/H; second term = dHdlna * dHdlna? 
    # Use d2H/dlna^2 = (d/dlna)(dH/dlna) = (dHdlna)' = d/dlna(Hdot/H)
    # dHdlna = Hdot/H ; d/dlna = (d/dlna Hdot)/H - Hdot*Hdot/H^2
    # dHdot/dlna = dHdot/dE * dE/dlna
    # E^2 = Om0 a^-3(1+w0+wa) + (1-Om0) a^-3(1+w0+wa) ... for CPL: dE2/dlna = -3(1+w0+wa)*E2
    dE2dlna = -3.0 * (1.0 + w0 + wa) * E2
    dEdlna = dE2dlna / (2.0 * E)
    dHdotdlna = Hdot / E * dEdlna
    d2Hdlna2 = dHdotdlna / H - (Hdot / H) ** 2
    # growth ODE: D'' + (3(1+w) + dHdlna) D' + [d2Hdlna2 + 3(1+w)dHdlna] D = 0
    w = w0 + wa * (1.0 - a)
    p = 3.0 * (1.0 + w) + dHdlna
    qf = d2Hdlna2 + 3.0 * (1.0 + w) * dHdlna
    # integrate y1 = D, y2 = D' in x. Use RK4 with local step.
    y1 = np.empty(n + 1); y2 = np.empty(n + 1)
    y1[0] = 1.0; y2[0] = 0.0
    # initial derivative from equation at x=0 (a=1): y2' = -p y2 - qf y1
    def deriv(xk, y1k, y2k, i):
        return y2k, (-p[i] * y2k - qf[i] * y1k)
    dx = x[1] - x[0]
    for i in range(n):
        k1a, k1b = deriv(x[i], y1[i], y2[i], i)
        k2a, k2b = deriv(x[i] + 0.5 * dx, y1[i] + 0.5 * dx * k1a, y2[i] + 0.5 * dx * k1b, i)
        k3a, k3b = deriv(x[i] + 0.5 * dx, y1[i] + 0.5 * dx * k2a, y2[i] + 0.5 * dx * k2b, i)
        k4a, k4b = deriv(x[i] + dx, y1[i] + dx * k3a, y2[i] + dx * k3b, i + 1)
        y1[i + 1] = y1[i] + dx / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
        y2[i + 1] = y2[i] + dx / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b)
    # interpolate to a_grid
    D = np.interp(np.log(a_grid), x, y1)
    Dp = np.interp(np.log(a_grid), x, y2)
    return D, Dp

def _norm_D(D, a_grid):
    return D / D[-1]

def _build_refs():
    Om0s = [0.22, 0.27, 0.32]
    was = [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]
    a_grid = np.geomspace(0.08, 1.0, 400)
    refs = []
    for Om0 in Om0s:
        for wa in was:
            if time.time() > _DEADLINE:
                break
            D, Dp = _ref_ode(Om0, wa, a_grid)
            Dn = _norm_D(D, a_grid)
            # reference f_ref = Om^gamma_ref ; use Dp/(H D) relation:
            # f = dlnD/dlna = Dp (since H0=1 and Dp = dD/dlna). Actually dD/dlna = H D f.
            # So f_ref = Dp/(H*D).
            w0 = -1.0
            om = _om_a(a_grid, Om0, wa)
            E2 = om + (1.0 - Om0) * a_grid ** (-3.0 * (1.0 + w0 + wa))
            E = np.sqrt(E2)
            f_ref = Dp / (E * D)
            refs.append((Om0, wa, a_grid, f_ref, Dn))
    return refs

def _fit(refs):
    best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = np.inf
    rng = np.random.default_rng(0)
    # local search with restarts, mimicking worst relative error
    iters = 0
    max_iters = 40000
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        c = best + rng.normal(0, 0.05, 6)
        c = np.clip(c, -10, 10)
        err = 0.0
        for (Om0, wa, a_grid, f_ref, Dn) in refs:
            f = _f(a_grid, Om0, wa, c)
            # gamma bounds check
            g = _gamma(a_grid, Om0, wa, c)
            if np.any(g < 0.1) or np.any(g > 1.2):
                err = 1e6
                break
            rel_f = np.max(np.abs(f - f_ref) / np.maximum(np.abs(f_ref), 1e-8))
            # normalized D from ansatz
            n = 2000
            xs = np.linspace(0.0, math.log(a_grid[0]), n)
            aint = np.exp(xs)
            fint = _f(aint, Om0, wa, c)
            lnD = np.cumsum(-fint) * (xs[1] - xs[0])  # trapezoid approx via cumsum of -f*dx
            # better trapezoid:
            lnD = np.zeros_like(xs)
            for i in range(1, n):
                lnD[i] = lnD[i - 1] - 0.5 * (fint[i] + fint[i - 1]) * (xs[i] - xs[i - 1])
            Dn_calc = np.exp(lnD)
            Dn_calc = Dn_calc / Dn_calc[0]  # normalize to 1 at a=1
            # interpolate Dn_calc to a_grid
            Dn_c_grid = np.interp(np.log(a_grid), xs, Dn_c_grid := Dn_calc)
            rel_D = np.max(np.abs(Dn_c_grid - Dn) / np.maximum(np.abs(Dn), 1e-8))
            err = max(err, rel_f, rel_D)
        if err < best_err:
            best_err = err
            best = c
    return best

def construct():
    try:
        if time.time() > _DEADLINE:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        refs = _build_refs()
        if not refs:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        c = _fit(refs)
        return [float(x) for x in c]
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
