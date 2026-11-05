"""Optimized compact growth-index approximation for CPL dark-energy cosmologies."""
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    """Return the six finite coefficients in [-10,10] for the growth-index ansatz.

    We optimize these offline (with a hard deadline guard) against a
    high-fidelity general-relativistic linear-growth ODE reference across a
    range of CPL cosmologies, then return the best coefficients found.
    """
    t0 = time.time()
    DEADLINE = 60.0
    MARGIN = 5.0

    # ---- cosmologies to fit: (Om0, Ode0, w0, wa) ----
    COSMO = [
        (0.30, 0.70, -1.0, 0.0),   # Planck-like LCDM (seed)
        (0.30, 0.70, -0.5, 0.0),
        (0.30, 0.70, -1.5, 0.0),
        (0.30, 0.70, -1.0, 0.5),
        (0.30, 0.70, -1.0, -0.5),
        (0.30, 0.70, -0.8, 0.3),
        (0.25, 0.75, -1.0, 0.0),
        (0.35, 0.65, -1.2, 0.2),
    ]

    # ---- a grid (log-spaced, includes a=1) ----
    a_grid = np.unique(np.concatenate([
        np.array([1.0]),
        np.logspace(np.log10(0.005), 0.0, 260),
        np.linspace(0.9, 1.0, 21),
    ]))

    G = 6.6743e-11
    c = 2.99792458e8
    rho_crit = 3 * (c / (299792458.0 * 1e-9)) ** 0  # placeholder
    # rho_crit in SI: 3 c^2 / (8 pi G)
    rho_crit = 3 * c ** 2 / (8 * np.pi * G)

    # precompute per-cosmology background arrays
    bg = []
    for (Om0, Ode0, w0, wa) in COSMO:
        a = a_grid
        Om = Om0 * a ** -3
        w = w0 + wa * (1 - a)
        ode = Ode0 * a ** -3 * np.exp(-3 * (w0 + wa / 2) * (1 - a))
        q = 1 - Om
        u = 1 + w
        E2 = Om + ode
        rho = rho_crit * E2
        H = c * np.sqrt(rho / (3 * rho_crit))  # = c * sqrt(E2)
        H2 = c ** 2 * E2
        bg.append(dict(a=a, Om=Om, q=q, u=u, E2=E2, rho=rho, H=H, H2=H2))

    # ---- reference: solve growth ODE on the a-grid ----
    # D'' + (3+q) H D' - 4 pi G rho D = 0  (primes = d/da)
    # integrate y=[D, D'] over a from a0 to 1
    A0 = 0.005
    n = len(a_grid)
    ref_f = np.empty((n, len(COSMO)))
    ref_D = np.empty((n, len(COSMO)))
    for ci, b in enumerate(bg):
        a = b['a']; H = b['H']; rho = b['rho']; q = b['q']
        i0 = np.searchsorted(a, A0)
        aa = a[i0:]
        Hs = H[i0:]; rs = rho[i0:]; qs = q[i0:]
        da = np.diff(aa)
        D = np.empty_like(aa); Dp = np.empty_like(aa)
        D[0] = 1.0; Dp[0] = 0.9
        for i in range(1, len(aa)):
            h = da[i - 1]
            k1d = Dp[i - 1]
            k1f = -((3 + qs[i - 1]) * Hs[i - 1] * Dp[i - 1] - 4 * np.pi * G * rs[i - 1] * D[i - 1])
            k2d = Dp[i - 1] + 0.5 * h * k1f
            k2f = -((3 + qs[i - 1]) * Hs[i - 1] * (Dp[i - 1] + 0.5 * h * k1f)
                    - 4 * np.pi * G * rs[i - 1] * (D[i - 1] + 0.5 * h * k1d))
            k3d = Dp[i - 1] + 0.5 * h * k2f
            k3f = -((3 + qs[i - 1]) * Hs[i - 1] * (Dp[i - 1] + 0.5 * h * k2f)
                    - 4 * np.pi * G * rs[i - 1] * (D[i - 1] + 0.5 * h * k3d))
            k4d = Dp[i - 1] + h * k3f
            k4f = -((3 + qs[i - 1]) * Hs[i - 1] * (Dp[i - 1] + h * k3f)
                    - 4 * np.pi * G * rs[i - 1] * (D[i - 1] + h * k4d))
            D[i] = D[i - 1] + h / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
            Dp[i] = Dp[i - 1] + h / 6.0 * (k1f + 2 * k2f + 2 * k3f + k4f)
        # map back to full grid (fill pre-A0 by back-extrapolation of f)
        f_full = np.empty(n)
        f_full[i0:] = Dp[i0:] / D[i0:]
        f_full[:i0] = f_full[i0]
        D_full = np.empty(n)
        D_full[i0:] = D[i0:]
        # integrate D backwards for pre-A0 (f ~ const)
        for i in range(i0 - 1, -1, -1):
            D_full[i] = D_full[i + 1] * np.exp(-f_full[i] * (a[i + 1] - a[i]))
        D_full[0] = 1.0
        D_full = D_full / D_full[-1]  # normalize D(1)=1
        ref_f[:, ci] = f_full
        ref_D[:, ci] = D_full

    # ---- ansatz functions ----
    def gamma_of(c, b):
        q = b['q']; u = b['u']; wa = COSMO[0][3]  # placeholder, replaced below
        return (c[0] + c[1] * q + c[2] * q ** 2
                + c[3] * u + c[4] * b['wa'] * b['a'] + c[5] * u * q)

    # store wa in bg
    for ci, b in enumerate(bg):
        b['wa'] = COSMO[ci][3]

    def f_ansatz(c, b):
        g = (c[0] + c[1] * b['q'] + c[2] * b['q'] ** 2
             + c[3] * b['u'] + c[4] * b['wa'] * b['a'] + c[5] * b['u'] * b['q'])
        return np.power(b['Om'], g)

    def D_ansatz(c, b):
        g = (c[0] + c[1] * b['q'] + c[2] * b['q'] ** 2
             + c[3] * b['u'] + c[4] * b['wa'] * b['a'] + c[5] * b['u'] * b['q'])
        f = np.power(b['Om'], g)
        # D/D(1) = exp(-int_a^1 f dx), x=ln a
        ln_a = np.log(b['a'])
        # cumulative integral from a to 1 of f dx = -(cumint_f[1]-cumint_f[i])
        # integrate f * dln_a = f / a da ; do cumulative trapezoid on x=ln_a
        integrand = f  # per unit x
        x = ln_a
        cum = np.empty(n)
        cum[-1] = 0.0
        # from right to left: cum[i] = int_{x_i}^{x_1} f dx
        for i in range(n - 2, -1, -1):
            cum[i] = cum[i + 1] + 0.5 * (integrand[i] + integrand[i + 1]) * (x[i + 1] - x[i])
        D = np.exp(-cum)
        return D

    def residual(c):
        r = []
        for ci, b in enumerate(bg):
            fa = f_ansatz(c, b)
            Da = D_ansatz(c, b)
            r.append(fa - ref_f[:, ci])
            r.append(Da - ref_D[:, ci])
        return np.concatenate(r)

    # ---- gamma feasibility check ----
    def gamma_ok(c):
        for b in bg:
            g = (c[0] + c[1] * b['q'] + c[2] * b['q'] ** 2
                 + c[3] * b['u'] + c[4] * b['wa'] * b['a'] + c[5] * b['u'] * b['q'])
            if np.min(g) < 0.1001 or np.max(g) > 1.1999:
                return False
        return True

    def cost(c):
        if not gamma_ok(c):
            return 1e6
        return float(np.max(np.abs(residual(c))))

    # ---- optimize ----
    best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_cost = cost(best)
    try:
        res = least_squares(residual, best, method='trf',
                            bounds=([-10, -10, -10, -10, -10, -10],
                                   [10, 10, 10, 10, 10, 10]),
                            max_nfev=400, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if gamma_ok(res.x):
            cc = cost(res.x)
            if cc < best_cost:
                best_cost = cc; best = res.x
    except Exception:
        pass

    if time.time() - t0 > DEADLINE - MARGIN:
        return [float(x) for x in best]

    # ---- local refinement: Nelder-Mead on the max-error cost ----
    try:
        from scipy.optimize import minimize
        x0 = best.copy()
        res2 = minimize(cost, x0, method='Nelder-Mead',
                        options=dict(maxiter=300, xatol=1e-8, fatol=1e-10))
        if gamma_ok(res2.x):
            cc = cost(res2.x)
            if cc < best_cost:
                best_cost = cc; best = res2.x
    except Exception:
        pass

    if time.time() - t0 > DEADLINE - MARGIN:
        return [float(x) for x in best]

    # ---- multi-start polish from a few perturbations ----
    rng = np.random.default_rng(0)
    for k in range(8):
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        x0 = best + rng.normal(0, 0.05, 6)
        x0 = np.clip(x0, -10, 10)
        try:
            res3 = least_squares(residual, x0, method='trf',
                                 bounds=([-10, -10, -10, -10, -10, -10],
                                         [10, 10, 10, 10, 10, 10]),
                                 max_nfev=200, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            if gamma_ok(res3.x):
                cc = cost(res3.x)
                if cc < best_cost:
                    best_cost = cc; best = res3.x
        except Exception:
            pass

    # final clamp to bounds
    best = np.clip(best, -10, 10)
    return [float(x) for x in best]
