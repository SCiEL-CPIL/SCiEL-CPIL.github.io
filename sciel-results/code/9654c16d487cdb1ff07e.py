"""Growth-index approximation: runtime fit of 6 coefficients against ODE reference."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _Om(a, Om0, w0, wa):
    """Matter density fraction Om(a) for CPL dark energy, normalized to a=1."""
    a = np.asarray(a, dtype=float)
    x = 1.0 - a
    # E(a)^2 = Om_m a^-3 + (1-Om_m) a^-3(1+x)^{3(1+w0+wa)}
    expo = 3.0 * (1.0 + w0 + wa)
    E2 = Om0 * a ** (-3.0) + (1.0 - Om0) * a ** (-3.0) * (1.0 + x) ** expo
    return Om0 / E2

def _gamma(c, a, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q ** 2 + c3 * u + c4 * wa * a + c5 * u * q

def _f_ode(a, Om0, w0, wa):
    """Reference f = d ln D/d ln a from the second-order growth ODE."""
    a0, a1 = 0.02, 1.0
    a = np.clip(np.asarray(a, dtype=float), a0, a1)
    n = 2001
    aa = np.linspace(a0, a1, n)
    H2 = _Om(aa, Om0, w0, wa) + (1.0 - _Om(aa, Om0, w0, wa))  # = 1/E^2
    H2 = np.clip(H2, 1e-12, None)
    H2p = np.gradient(H2, aa)
    H2pp = np.gradient(H2p, aa)
    Om = _Om(aa, Om0, w0, wa)

    # y = [D, dD/dlna]; D' = v, v' = -1.5 v - (1.5 Om - 0.5 H2'/H2) v - (0.5 H2''/H2 - 0.75 H2'/H2) D
    def rhs(lna, y):
        i = int(round((lna - np.log(a0)) / (np.log(a1) - np.log(a0)) * (n - 1)))
        i = min(max(i, 0), n - 1)
        h2 = H2[i]
        h2p = H2p[i]
        h2pp = H2pp[i]
        om = Om[i]
        D, v = y
        dv = -1.5 * v + (0.5 * h2p / h2 - 1.5 * om) * v + (0.5 * h2pp / h2 - 0.75 * h2p / h2) * D
        return [v, dv]

    # Integrate forward from a0 with standard initial conditions
    D0 = 1.0
    v0 = Om[a0] ** 0.55  # reasonable seed
    lna_grid = np.log(np.linspace(a0, a1, 4001))
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, [np.log(a0), np.log(a1)], [D0, v0],
                    t_eval=lna_grid, method="RK45", rtol=1e-9, atol=1e-12)
    if not sol.success:
        # fallback: use power-law reference
        Om_a = _Om(a, Om0, w0, wa)
        return Om_a ** 0.55
    D = sol.y[0]
    v = sol.y[1]
    f = np.interp(np.log(a), lna_grid, v)
    return f

def _D_ode(a, Om0, w0, wa):
    """Reference normalized growth factor D(a)/D(1) via ODE."""
    a0, a1 = 0.02, 1.0
    a = np.clip(np.asarray(a, dtype=float), a0, a1)
    n = 2001
    aa = np.linspace(a0, a1, n)
    H2 = _Om(aa, Om0, w0, wa) + (1.0 - _Om(aa, Om0, w0, wa))
    H2 = np.clip(H2, 1e-12, None)
    H2p = np.gradient(H2, aa)
    H2pp = np.gradient(H2p, aa)
    Om = _Om(aa, Om0, w0, wa)

    def rhs(lna, y):
        i = int(round((lna - np.log(a0)) / (np.log(a1) - np.log(a0)) * (n - 1)))
        i = min(max(i, 0), n - 1)
        h2 = H2[i]; h2p = H2p[i]; h2pp = H2pp[i]; om = Om[i]
        D, v = y
        dv = -1.5 * v + (0.5 * h2p / h2 - 1.5 * om) * v + (0.5 * h2pp / h2 - 0.75 * h2p / h2) * D
        return [v, dv]

    from scipy.integrate import solve_ivp
    D0 = 1.0
    v0 = Om[a0] ** 0.55
    lna_grid = np.log(np.linspace(a0, a1, 4001))
    sol = solve_ivp(rhs, [np.log(a0), np.log(a1)], [D0, v0],
                    t_eval=lna_grid, method="RK45", rtol=1e-9, atol=1e-12)
    if not sol.success:
        Om_a = _Om(a, Om0, w0, wa)
        return np.exp(-0.55 * (np.log(1.0) - np.log(a)))  # D ~ a^0.55
    D = sol.y[0]
    Dnorm = D / D[-1]
    D_a = np.interp(np.log(a), lna_grid, Dnorm)
    return D_a

def construct():
    """Return six coefficients for the compact growth-index ansatz, fitted at runtime."""
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin

    # Grid of cosmologies and scale factors (biased toward low a where error peaks)
    Om0s = [0.22, 0.27, 0.30, 0.35]
    w0s = [-1.2, -1.0, -0.8]
    was = [-0.3, -0.1, 0.0, 0.1, 0.3]
    a_vals = np.concatenate([
        np.linspace(0.03, 0.15, 12),
        np.linspace(0.15, 0.5, 10),
        np.linspace(0.5, 1.0, 8),
    ])

    # Build reference data
    ref_f = []
    ref_D = []
    ref_a = []
    ref_Om0 = []
    ref_w0 = []
    ref_wa = []
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                f_ref = _f_ode(a_vals, Om0, w0, wa)
                D_ref = _D_ode(a_vals, Om0, w0, wa)
                # skip non-finite
                mask = np.isfinite(f_ref) & np.isfinite(D_ref)
                for j in range(len(a_vals)):
                    if not mask[j]:
                        continue
                    ref_f.append(f_ref[j])
                    ref_D.append(D_ref[j])
                    ref_a.append(a_vals[j])
                    ref_Om0.append(Om0)
                    ref_w0.append(w0)
                    ref_wa.append(wa)

    ref_f = np.array(ref_f)
    ref_D = np.array(ref_D)
    ref_a = np.array(ref_a)
    ref_Om0 = np.array(ref_Om0)
    ref_w0 = np.array(ref_w0)
    ref_wa = np.array(ref_wa)

    npts = len(ref_f)

    def _relerr(c):
        if time.time() > deadline - 3.0:
            return 1e9
        c = np.clip(c, -10.0, 10.0)
        Om = _Om(ref_a, ref_Om0, ref_w0, ref_wa)
        q = 1.0 - Om
        u = 1.0 + ref_w0 + ref_wa * (1.0 - ref_a)
        g = c[0] + c[1] * q + c[2] * q ** 2 + c[3] * u + c[4] * ref_wa * ref_a + c[5] * u * q
        # gamma constraint [0.1, 1.2]
        g_pen = np.maximum(0.0, 0.1 - g) + np.maximum(0.0, g - 1.2)
        f_app = np.exp(g * np.log(Om))
        rf = np.abs(f_app - ref_f) / np.maximum(ref_f, 1e-12)
        # D via exp(-int f dlna): approximate by cumulative trapezoid on log a
        # Build per-cosmology grouping for integration
        # To keep it cheap, integrate over the full ref_a (assumed sorted per cosmology block)
        # We'll compute D_app by integrating f_app over ln a using cumulative trapezoid.
        ln_a = np.log(np.maximum(ref_a, 1e-12))
        # cumulative integral of f_app dlna from a to 1 (ref_a increasing)
        # We need integral from ln a to ln 1 = 0. Since ref_a <= 1, integral over [ln a, 0].
        # Approximate: total integral over [ln_a[0], 0] minus integral over [ln_a[0], ln_a[i]].
        # Build cumulative trapezoid over sorted unique ln_a.
        # Since ref_a is non-decreasing, we can do cumulative directly.
        # integral_i = ∫_{ln a_i}^{0} f dlna
        # Use cumulative from the end.
        # trapezoid of f_app over ln_a
        dln = np.diff(ln_a)
        trap = 0.5 * (f_app[:-1] + f_app[1:]) * dln
        cum_from_start = np.concatenate([[0.0], np.cumsum(trap)])
        total = cum_from_start[-1]
        integral_i = total - cum_from_start  # from ln_a[i] to ln_a[-1] ~ 0
        # but ln_a[-1] may be < 0 (a<1). Add tail from ln_a[-1] to 0 with f at end.
        tail = f_app[-1] * (0.0 - ln_a[-1])
        integral_i = integral_i + tail
        D_app = np.exp(-integral_i)
        rD = np.abs(D_app - ref_D) / np.maximum(ref_D, 1e-12)
        err = np.maximum(rf, rD)
        # robust loss: max + small mean to guide gradient
        loss = np.max(err) + 0.01 * np.mean(err) + 5.0 * np.mean(g_pen)
        return loss

    # Initial guess: extend seed with small extra terms
    c0_init = np.array([0.535, 0.068, -0.012, 0.0, 0.0, 0.0])

    best_c = c0_init.copy()
    best_loss = _relerr(best_c)

    # Try Nelder-Mead (robust, no gradients)
    try:
        res = minimize(_relerr, best_c, method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-5,
                                "adaptive": True})
        if res.success or res.fun < best_loss:
            if res.fun < best_loss:
                best_c = res.x
                best_loss = res.fun
    except Exception:
        pass

    # Refine with L-BFGS-B (gradient-based) from best
    try:
        res2 = minimize(_relerr, best_c, method="L-BFGS-B",
                        bounds=[(-10, 10)] * 6,
                        options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-5})
        if res2.fun < best_loss:
            best_c = res2.x
            best_loss = res2.fun
    except Exception:
        pass

    # Final safety: clip and ensure finite
    best_c = np.clip(best_c, -10.0, 10.0)
    best_c = np.nan_to_num(best_c, nan=0.0, posinf=10.0, neginf=-10.0)
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
