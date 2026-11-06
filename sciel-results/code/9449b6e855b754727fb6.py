"""Compact dark-energy growth-index fit; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 82.0  # safety margin below the 90 s wall-clock kill


def _now():
    return time.time()


def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    wa_a = 1.0 - a
    w = w0 + wa * wa_a
    x = np.maximum(a, 1e-300)
    num = 1.0 + w
    den = 1.0 + 3.0 * w
    E2 = Om0 * np.power(x, 3.0 * den) + (1.0 - Om0) * np.power(x, -3.0 * num)
    return Om0 * np.power(x, 3.0) / E2


def _f_gr(a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    a = np.maximum(np.asarray(a, dtype=float), 1e-300)
    dOm = (3.0 * Om - 1.5 * Om * (w0 + wa * (1.0 - a))) / a
    f = 0.5 * Om + 0.6 * np.sqrt(np.maximum(Om * (1.0 - Om), 0.0)) + 0.2 * dOm / (3.0 * a)
    return f


def _D_gr(a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    f = _f_gr(np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _gamma(c, a, Om0, w0, wa):
    q = 1.0 - _Om(a, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _f_app(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    g = _gamma(c, a, Om0, w0, wa)
    return np.power(np.maximum(Om, 1e-300), g)


def _D_app(c, a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    fa = _f_app(c, np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (fa[1:] + fa[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _eval(c, models, a_eval, a_fit):
    # Check gamma bounds
    for (Om0, w0, wa) in models:
        g = _gamma(c, a_fit, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1e9
        if not np.all(np.isfinite(g)):
            return 1e9
    
    err_f = 0.0
    err_D = 0.0
    for (Om0, w0, wa) in models:
        fa = _f_app(c, a_eval, Om0, w0, wa)
        fg = _f_gr(a_eval, Om0, w0, wa)
        err_f = max(err_f, float(np.max(np.abs((fa - fg) / np.maximum(fg, 1e-10)))))
        
        Da = _D_app(c, a_eval, Om0, w0, wa)
        Dg = _D_gr(a_eval, Om0, w0, wa)
        err_D = max(err_D, float(np.max(np.abs((Da - Dg) / np.maximum(Dg, 1e-10)))))
    
    return 100.0 * max(err_f, err_D)


def construct():
    t0 = _now()
    # Expanded set of models for robustness as per strategy
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 1.0),
        (0.30, -1.0, 0.5),
        (0.27, -1.0, 0.3),
        (0.30, -0.9, 0.0),
        (0.315, -1.0, -0.5),
        (0.25, -1.0, 1.5),
        (0.30, -1.2, 0.8),
        (0.28, -0.7, 0.4),
        (0.32, -1.1, -0.3),
    ]
    a_eval = np.logspace(-3.0, 0.0, 150)
    a_fit = np.logspace(-3.0, 0.0, 100)

    # Pre-compute reference data to speed up objective function
    # We store f_gr and D_gr for each model and a_eval point
    ref_data = []
    for m in models:
        Om0, w0, wa = m
        fg = _f_gr(a_eval, Om0, w0, wa)
        Dg = _D_gr(a_eval, Om0, w0, wa)
        # Store as arrays for vectorized evaluation
        ref_data.append((fg, Dg))

    def _fast_eval(c):
        """Vectorized evaluation against pre-computed references."""
        c = np.clip(c, -10.0, 10.0)
        # Check gamma bounds quickly on a subset of a_fit to save time in inner loop
        # Actually, we should check on a_fit to be safe, but a_eval is used for error.
        # The contract says gamma must be in [0.1, 1.2] at all checked locations.
        # We assume a_fit covers the range of interest.
        max_err = 0.0
        for i, (Om0, w0, wa) in enumerate(models):
            # Calculate gamma for a_fit to check bounds
            q_fit = 1.0 - _Om(a_fit, Om0, w0, wa)
            u_fit = 1.0 + w0 + wa * (1.0 - a_fit)
            g_fit = c[0] + c[1]*q_fit + c[2]*q_fit**2 + c[3]*u_fit + c[4]*wa*a_fit + c[5]*u_fit*q_fit
            if np.any(g_fit < 0.1) or np.any(g_fit > 1.2):
                return 1e9
            
            # Calculate f_app and D_app for a_eval
            fa = _f_app(c, a_eval, Om0, w0, wa)
            Da = _D_app(c, a_eval, Om0, w0, wa)
            
            fg, Dg = ref_data[i]
            
            # Relative error for f
            err_f = np.abs(fa - fg) / np.maximum(fg, 1e-10)
            # Relative error for D
            err_D = np.abs(Da - Dg) / np.maximum(Dg, 1e-10)
            
            m_err = max(np.max(err_f), np.max(err_D))
            if m_err > max_err:
                max_err = m_err
        
        return 100.0 * max_err

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_v = _fast_eval(best_c)

    def _try(c):
        nonlocal best_c, best_v
        if not np.all(np.isfinite(c)):
            return
        v = _fast_eval(c)
        if v < best_v:
            best_v = v
            best_c = np.clip(c, -10.0, 10.0)

    from scipy.optimize import differential_evolution, minimize

    # 1. Differential Evolution for global search with tighter bounds
    if _now() < _DEADLINE - 25.0:
        try:
            # Use tighter initial bounds based on physical expectations
            bounds = [
                (-1.0, 2.0),   # c0: base gamma ~0.55
                (-2.0, 2.0),   # c1: q coefficient
                (-2.0, 2.0),   # c2: q^2 coefficient
                (-2.0, 2.0),   # c3: u coefficient
                (-2.0, 2.0),   # c4: wa*a coefficient
                (-2.0, 2.0),   # c5: u*q coefficient
            ]
            de_res = differential_evolution(
                _fast_eval,
                bounds,
                maxiter=80,
                popsize=20,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol'
            )
            _try(de_res.x)
        except Exception:
            pass

    # 2. Multi-start local optimization
    start_points = [best_c.copy(), np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])]
    
    # Add random perturbations around best
    rng = np.random.default_rng(42)
    for _ in range(8):
        start_points.append(np.clip(best_c + rng.normal(0, 0.15, 6), -10, 10))
    
    # Add some structured starts
    start_points.append(np.array([0.5, 0.1, -0.05, 0.05, 0.02, -0.01]))
    start_points.append(np.array([0.6, -0.1, 0.05, -0.03, 0.01, 0.02]))

    for sp in start_points:
        if _now() > _DEADLINE - 5.0:
            break
        try:
            # Try L-BFGS-B first (faster, uses gradient approximation)
            lbfgs_res = minimize(
                _fast_eval,
                sp,
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0)] * 6,
                options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8}
            )
            _try(lbfgs_res.x)
        except Exception:
            pass
        
        if _now() > _DEADLINE - 5.0:
            break
        try:
            # Then Nelder-Mead for robustness
            nm_res = minimize(
                _fast_eval,
                sp,
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-9, "fatol": 1e-10}
            )
            _try(nm_res.x)
        except Exception:
            pass

    # Final validation
    best_c = np.clip(best_c, -10.0, 10.0)
    # Double check gamma bounds on a_fit for all models
    gamma_ok = True
    for (Om0, w0, wa) in models:
        q_fit = 1.0 - _Om(a_fit, Om0, w0, wa)
        u_fit = 1.0 + w0 + wa * (1.0 - a_fit)
        g_fit = best_c[0] + best_c[1]*q_fit + best_c[2]*q_fit**2 + best_c[3]*u_fit + best_c[4]*wa*a_fit + best_c[5]*u_fit*q_fit
        if np.any(g_fit < 0.1) or np.any(g_fit > 1.2) or not np.all(np.isfinite(g_fit)):
            gamma_ok = False
            break
    
    if not gamma_ok:
        # Fallback to safe seed
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Final polish with very tight tolerances if time permits
    if _now() < _DEADLINE - 3.0 and gamma_ok:
        try:
            final_res = minimize(
                _fast_eval,
                best_c,
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0)] * 6,
                options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10}
            )
            fc = np.clip(final_res.x, -10.0, 10.0)
            fv = _fast_eval(fc)
            if fv < best_v:
                # Verify gamma bounds for final candidate
                final_ok = True
                for (Om0, w0, wa) in models:
                    q_fit = 1.0 - _Om(a_fit, Om0, w0, wa)
                    u_fit = 1.0 + w0 + wa * (1.0 - a_fit)
                    g_fit = fc[0] + fc[1]*q_fit + fc[2]*q_fit**2 + fc[3]*u_fit + fc[4]*wa*a_fit + fc[5]*u_fit*q_fit
                    if np.any(g_fit < 0.1) or np.any(g_fit > 1.2) or not np.all(np.isfinite(g_fit)):
                        final_ok = False
                        break
                if final_ok:
                    best_c = fc
                    best_v = fv
        except Exception:
            pass

    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
