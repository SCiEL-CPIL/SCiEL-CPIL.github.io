"""Growth-index fitting: fit six compact ansatz coefficients to GR reference."""
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _Om_m(a, Om0, wa):
    """Matter density factor Om(a) = Om0 * a^-3 / E^2."""
    x = a ** 3
    y = a ** (-3.0 * wa)
    E2 = Om0 * x + (1.0 - Om0) * y
    return Om0 * x / E2

def _H2(a, Om0, wa):
    """E^2(a) = H^2/H0^2."""
    x = a ** 3
    y = a ** (-3.0 * wa)
    return Om0 * x + (1.0 - Om0) * y

def _ref_f(a, Om0, wa):
    """True growth rate f = Om(a) * (1/2 + 3/2 * Om(a) - 1/(6H) dH/dlna)."""
    E2 = _H2(a, Om0, wa)
    Om = _Om_m(a, Om0, wa)
    # dE2/dlna = 3*Om0*a^3 - 3*wa*(1-Om0)*a^(-3wa)
    dE2 = 3.0 * Om0 * a**3 - 3.0 * wa * (1.0 - Om0) * a**(-3.0 * wa)
    dlnH = 0.5 * dE2 / E2
    f = Om * (1.0 + 1.5 * Om - dlnH)
    return f

def _ref_D(a, Om0, wa):
    """Normalized growth factor D(a)/D(1) = exp(-int_0^a f/E da/a)."""
    # Integrate in log space for stability
    a = np.asarray(a, dtype=float)
    # Use substitution x = ln(a), integrate from 0 to ln(a)
    # f(a)/E(a) * da/a = f(a)/E(a) dx
    def integrand(x, a_arr):
        aa = np.exp(x)
        ff = _ref_f(aa, Om0, wa)
        EE = np.sqrt(_H2(aa, Om0, wa))
        return ff / EE
    # For a single or array
    if np.isscalar(a) or a.ndim == 0:
        a = np.array([a])
    result = np.zeros_like(a)
    for i, ai in enumerate(a):
        if ai <= 0:
            result[i] = 0.0
        elif ai >= 1.0:
            result[i] = 1.0
        else:
            x_end = np.log(ai)
            sol = solve_ivp(integrand, [0.0, x_end], [0.0],
                           method='RK45', rtol=1e-10, atol=1e-12,
                           args=(a,))
            result[i] = np.exp(-sol.y[0, -1])
    if result.ndim == 0:
        return result[0]
    return result

def _gamma_model(c, q, u, wa, a):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _f_model(c, a, Om0, wa):
    """Compute f = Om(a)^gamma from the ansatz."""
    a = np.asarray(a, dtype=float)
    Om = _Om_m(a, Om0, wa)
    q = 1.0 - Om
    u = 1.0 + wa + wa*(1.0 - a)
    g = _gamma_model(c, q, u, wa, a)
    # Clamp gamma to valid range
    g = np.clip(g, 0.1, 1.2)
    f = Om ** g
    return f

def _D_model(c, a, Om0, wa):
    """Compute normalized D from the ansatz via quadrature."""
    a = np.asarray(a, dtype=float)
    if np.isscalar(a) or a.ndim == 0:
        a = np.array([a])
    result = np.zeros_like(a)
    for i, ai in enumerate(a):
        if ai <= 0:
            result[i] = 0.0
        elif ai >= 1.0:
            result[i] = 1.0
        else:
            x_end = np.log(ai)
            def integrand(x):
                aa = np.exp(x)
                Omaa = _Om_m(aa, Om0, wa)
                q = 1.0 - Omaa
                u = 1.0 + wa + wa*(1.0 - aa)
                g = _gamma_model(c, q, u, wa, aa)
                g = np.clip(g, 0.1, 1.2)
                f = Omaa ** g
                E = np.sqrt(_H2(aa, Om0, wa))
                return f / E
            sol = solve_ivp(integrand, [0.0, x_end], [0.0],
                           method='RK45', rtol=1e-8, atol=1e-10)
            result[i] = np.exp(-sol.y[0, -1])
    if result.ndim == 0:
        return result[0]
    return result

def _objective(c, models, a_grid):
    """Worst relative error across all models and a values for both f and D."""
    max_err = 0.0
    for Om0, wa in models:
        f_ref = _ref_f(a_grid, Om0, wa)
        f_mod = _f_model(c, a_grid, Om0, wa)
        # Relative error in f
        mask = np.abs(f_ref) > 1e-10
        if np.any(mask):
            err_f = np.max(np.abs((f_mod[mask] - f_ref[mask]) / f_ref[mask]))
            max_err = max(max_err, err_f)
        # Growth factor error
        D_ref = _ref_D(a_grid, Om0, wa)
        D_mod = _D_model(c, a_grid, Om0, wa)
        maskD = np.abs(D_ref) > 1e-10
        if np.any(maskD):
            err_D = np.max(np.abs((D_mod[maskD] - D_ref[maskD]) / D_ref[maskD]))
            max_err = max(max_err, err_D)
    return max_err

def construct():
    """Fit six coefficients to minimize worst-case error across CPL models."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Representative CPL models spanning validation space
    models = [
        (0.30, 0.0),   # LCDM
        (0.27, -0.5),  # CPL-like
        (0.31, 0.3),   # positive wa
        (0.25, -1.0),  # strong negative wa
        (0.35, 0.5),   # high Om0, positive wa
        (0.20, -0.8),  # low Om0
        (0.28, 0.0),   # near LCDM
        (0.32, -0.3),  # mild CPL
    ]

    # Grid of scale factors for fitting
    a_grid = np.linspace(0.05, 0.95, 20)

    # Initial guess: constant gamma ~ 0.55 with small corrections
    c0_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    best_c = c0_init
    best_err = _objective(best_c, models, a_grid)

    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        if i == 0:
            c_init = c0_init
        else:
            c_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            for j in range(6):
                c_init[j] += rng.uniform(-0.1, 0.1)

        try:
            res = minimize(_objective, c_init, args=(models, a_grid),
                         method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6,
                                  'adaptive': True})
            if res.success and res.fun < best_err:
                best_err = res.fun
                best_c = res.x
        except Exception:
            pass

    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)

    # Final refinement with tighter tolerance if time permits
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective, best_c, args=(models, a_grid),
                         method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8,
                                  'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x
                best_c = np.clip(best_c, -10.0, 10.0)
        except Exception:
            pass

    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
