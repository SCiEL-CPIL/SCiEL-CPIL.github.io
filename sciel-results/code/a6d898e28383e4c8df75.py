import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # ---------- Reference growth ODE (in x = ln a) ----------
    # y = D, y' = dD/dx.  f = y'/y.
    # d(y')/dx = -[3/2 + 1.5(1+w) + d(1+w)/dx] y' + (3/2) Om y
    # d(1+w)/dx = wa * d(1-a)/dx = wa * a  (since d(1-a)/dx = -a)

    def h2_of_a(a, Om, w0, wa):
        # H^2/H0^2 = Om a^-3 + (1-Om) exp(3*wa*(1-a))
        return Om * a ** (-3.0) + (1.0 - Om) * math.exp(3.0 * wa * (1.0 - a))

    def om_m_of_a(a, Om, w0, wa):
        return Om * a ** (-3.0) / h2_of_a(a, Om, w0, wa)

    def growth_rhs(x, y, yp, Om, w0, wa):
        a = math.exp(x)
        w = w0 + wa * (1.0 - a)
        dw_dx = wa * a  # d(1+w)/dx = wa*a
        Om_a = om_m_of_a(a, Om, w0, wa)
        dyp_dx = -(1.5 + 1.5 * (1.0 + w) + dw_dx) * yp + 1.5 * Om_a * y
        return (yp, dyp_dx)

    def integrate_growth(Om, w0, wa, xs):
        # RK4 from x0 to each x in xs (xs sorted ascending, starting at x0)
        n = len(xs)
        Y = np.empty((n, 2))
        Y[0, 0] = 1.0
        Y[0, 1] = 0.5
        for i in range(1, n):
            x0 = xs[i - 1]
            x1 = xs[i]
            h = x1 - x0
            y0, yp0 = Y[i - 1, 0], Y[i - 1, 1]
            k1y, k1yp = growth_rhs(x0, y0, yp0, Om, w0, wa)
            k2y, k2yp = growth_rhs(x0 + 0.5 * h, y0 + 0.5 * h * k1y, yp0 + 0.5 * h * k1yp, Om, w0, wa)
            k3y, k3yp = growth_rhs(x0 + 0.5 * h, y0 + 0.5 * h * k2y, yp0 + 0.5 * h * k2yp, Om, w0, wa)
            k4y, k4yp = growth_rhs(x0 + h, y0 + h * k3y, yp0 + h * k3yp, Om, w0, wa)
            Y[i, 0] = y0 + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            Y[i, 1] = yp0 + (h / 6.0) * (k1yp + 2 * k2yp + 2 * k3yp + k4yp)
        # f = y'/y ; D normalized: D/D(1) = Y[:,0]/Y[-1,0]
        f = Y[:, 1] / Y[:, 0]
        D = Y[:, 0] / Y[-1, 0]
        return f, D

    # ---------- Model ansatz ----------
    def model_f(c, a, Om, w0, wa):
        c0, c1, c2, c3, c4, c5 = c
        Om_a = om_m_of_a(a, Om, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
        return Om_a ** gamma

    def model_D(c, a_grid, Om, w0, wa):
        # D/D(1) = exp(-int_{ln a}^{0} f(x) dx)
        # integrate f over x from ln(a) to 0
        # Use trapezoid on the given a_grid (which is in a, convert to x)
        xs = np.log(a_grid)
        fs = model_f(c, a_grid, Om, w0, wa)
        # We need integral from x to 0 for each x in xs (xs <= 0)
        # Build cumulative integral from the end (x=0) backwards
        # xs is ascending (a ascending). x=0 at a=1 (last point).
        # integral from x_i to 0 = sum over segments from i to end
        dx = np.diff(xs)  # positive
        trap = 0.5 * (fs[:-1] + fs[1:]) * dx
        # cumulative from the end: cum[j] = integral from x_j to 0
        cum = np.zeros(len(xs))
        # cum[n-1] = 0
        for j in range(len(xs) - 2, -1, -1):
            cum[j] = cum[j + 1] + trap[j]
        D = np.exp(-cum)
        return D

    # ---------- Cosmology grid ----------
    Om_grid = np.array([0.20, 0.25, 0.27, 0.30, 0.35, 0.40])
    w0_grid = np.array([-1.5, -1.2, -1.0, -0.8, -0.5, 0.0])
    wa_grid = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    # a grid for evaluation
    a_eval = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0])
    # a grid for integration (denser)
    a_int = np.linspace(0.05, 1.0, 400)

    # Precompute reference for a subset of cosmologies (to keep time bounded)
    # We'll pick a representative subset for fitting, then validate on more.
    # For fitting, use a moderate grid.
    fit_cosmos = []
    for Om in [0.22, 0.27, 0.32, 0.38]:
        for w0 in [-1.2, -1.0, -0.7, -0.3, 0.0]:
            for wa in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                fit_cosmos.append((Om, w0, wa))

    # Precompute reference data for fitting cosmologies
    ref_data = {}
    for (Om, w0, wa) in fit_cosmos:
        if time.time() > DEADLINE:
            break
        f_ref, D_ref = integrate_growth(Om, w0, wa, np.log(a_int))
        ref_data[(Om, w0, wa)] = (f_ref, D_ref)

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---------- Smooth objective using log-sum-exp for max ----------
    def objective(c, beta=100.0):
        """Smooth max via log-sum-exp. beta controls sharpness."""
        c = np.asarray(c)
        all_errs = []
        for key, (f_ref, D_ref) in ref_data.items():
            Om, w0, wa = key
            f_mod = model_f(c, a_int, Om, w0, wa)
            D_mod = model_D(c, a_int, Om, w0, wa)
            # relative errors
            err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            all_errs.append(err_f)
            all_errs.append(err_D)
        
        if not all_errs:
            return 0.0
        
        # Stack all errors
        all_errs = np.concatenate(all_errs)
        
        # Log-sum-exp for smooth max
        m = np.max(all_errs)
        if m < 1e-12:
            return 0.0
        # Use scaled log-sum-exp to avoid overflow
        scaled = all_errs / m
        log_sum_exp = m + np.log(np.sum(np.exp(beta * (scaled - 1.0)))) / beta
        return float(log_sum_exp)

    def objective_max(c):
        """Exact max objective for final evaluation."""
        c = np.asarray(c)
        max_err = 0.0
        for key, (f_ref, D_ref) in ref_data.items():
            Om, w0, wa = key
            f_mod = model_f(c, a_int, Om, w0, wa)
            D_mod = model_D(c, a_int, Om, w0, wa)
            # relative errors
            err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            max_err = max(max_err, float(np.max(err_f)), float(np.max(err_D)))
        return max_err

    # ---------- Nelder-Mead optimization with smooth objective ----------
    from scipy.optimize import minimize

    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = x0.copy()
    best_val = objective_max(x0)

    # Try more diverse starting points
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.1, 0.0, -0.05, 0.0, 0.0],
        [0.5, -0.05, 0.05, 0.0, 0.05, 0.0],
        [0.6, 0.05, -0.02, 0.02, -0.03, 0.01],
    ]
    for s in starts:
        if time.time() > DEADLINE - 8.0:
            break
        s = np.asarray(s)
        try:
            # Use smooth objective for optimization
            res = minimize(lambda x: objective(x, beta=100.0), s, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'initial_simplex': None})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass

    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)

    # Final polish with more cosmologies if time permits
    if time.time() < DEADLINE - 12.0:
        # Add more cosmologies for final validation-based refinement
        extra_cosmos = []
        for Om in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for w0 in [-1.5, -1.0, -0.5, 0.0]:
                for wa in [-0.5, 0.0, 0.5]:
                    if (Om, w0, wa) not in ref_data:
                        extra_cosmos.append((Om, w0, wa))
        for (Om, w0, wa) in extra_cosmos:
            if time.time() > DEADLINE - 6.0:
                break
            try:
                f_ref, D_ref = integrate_growth(Om, w0, wa, np.log(a_int))
                ref_data[(Om, w0, wa)] = (f_ref, D_ref)
            except Exception:
                pass
        
        # Re-optimize with full set using smooth objective
        if time.time() < DEADLINE - 6.0:
            try:
                res = minimize(lambda x: objective(x, beta=100.0), best_c, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x
            except Exception:
                pass
        
        # High-precision polish pass with exact max objective
        if time.time() < DEADLINE - 3.0:
            try:
                res = minimize(objective_max, best_c, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x
            except Exception:
                pass
        
        best_c = np.clip(best_c, -10.0, 10.0)

    # Ensure gamma stays in [0.1, 1.2] roughly; if not, fall back to constant
    # Quick check: evaluate gamma over a grid
    try:
        min_g = 1.2
        max_g = 0.1
        for Om in [0.2, 0.4]:
            for w0 in [-1.5, 0.0]:
                for wa in [-0.5, 0.5]:
                    a_test = np.linspace(0.1, 1.0, 20)
                    Om_a = om_m_of_a(a_test, Om, w0, wa)
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a_test)
                    g = best_c[0] + best_c[1]*q + best_c[2]*q*q + best_c[3]*u + best_c[4]*wa*a_test + best_c[5]*u*q
                    min_g = min(min_g, float(np.min(g)))
                    max_g = max(max_g, float(np.max(g)))
        if min_g < 0.1 or max_g > 1.2:
            # Fall back to constant gamma
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    except Exception:
        pass

    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
