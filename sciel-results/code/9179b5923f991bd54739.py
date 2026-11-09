import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    # Cosmologies covering likely validation space
    cosmologies = [
        (0.3, -1.0, 0.0),    # LCDM
        (0.3, -1.0, 0.3),    # CPL positive wa
        (0.3, -1.0, -0.3),   # CPL negative wa
        (0.25, -1.0, 0.0),   # Low Om
        (0.35, -1.0, 0.0),   # High Om
        (0.3, -0.9, 0.1),    # General CPL
        (0.3, -0.8, -0.2),   # General CPL
        (0.3, -0.5, 0.5),    # Quintessence
        (0.3, -1.2, 0.5),    # Phantom
        (0.3, 0.0, 0.0),     # Matter-only
    ]

    # a values: log-spaced for good coverage
    a_vals = np.logspace(-3, 0, 60)

    # Compute reference data via ODE
    ref_data = []
    for om, w0, wa in cosmologies:
        if elapsed() > deadline:
            break
        try:
            def E_sq(a):
                de_exp = -3.0 * (1.0 + w0 + wa)
                return om * a**(-3.0) + (1.0 - om) * a**de_exp

            def dE_sq_da(a):
                de_exp = -3.0 * (1.0 + w0 + wa)
                term_m = -3.0 * om * a**(-4.0)
                term_de = de_exp * (1.0 - om) * a**(de_exp - 1.0)
                return term_m + term_de

            def ode(x, y):
                a = np.exp(x)
                D, Dp = y
                e2 = E_sq(a)
                de2 = dE_sq_da(a)
                if e2 <= 0:
                    return [0.0, 0.0]
                hph = 0.5 * de2 / e2
                om_a = om * a**(-3.0) / e2
                Dpp = 1.5 * om_a / (a * a) * D - (3.0 + hph) * Dp
                return [Dp, Dpp]

            x_min = np.log(1e-4)
            a0 = np.exp(x_min)
            y0 = [a0, 1.0]
            sol = solve_ivp(ode, [x_min, 0.0], y0, rtol=1e-10, atol=1e-12, dense_output=True)
            if not sol.success:
                continue

            D_arr = sol.sol(np.log(a_vals))[0]
            Dp_arr = sol.sol(np.log(a_vals))[1]
            D_norm = D_arr / D_arr[-1]
            f_arr = a_vals * Dp_arr / D_arr
            ref_data.append((om, w0, wa, D_norm, f_arr))
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute arrays for fast objective
    precomputed = []
    for om, w0, wa, D_ref, f_ref in ref_data:
        e2 = om * a_vals**(-3.0) + (1.0 - om) * a_vals**(-3.0 * (1.0 + w0 + wa))
        om_a = om * a_vals**(-3.0) / e2
        q = 1.0 - om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        x_vals = np.log(a_vals)
        dx = np.diff(x_vals)
        precomputed.append((om_a, q, u, wa, x_vals, dx, D_ref, f_ref))

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        for om_a, q, u, wa, x_vals, dx, D_ref, f_ref in precomputed:
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            f_app = om_a ** gamma
            f_avg = 0.5 * (f_app[:-1] + f_app[1:])
            int_steps = f_avg * dx
            int_cum = np.concatenate(([0.0], np.cumsum(int_steps)))
            total_int = int_cum[-1]
            exponents = -(total_int - int_cum)
            D_app = np.exp(exponents)
            err_f = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            err_D = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
            local_max = max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max
        return max_err

    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_coeffs = x0.copy()
    best_score = objective(x0)

    bounds = [(-10, 10)] * 6

    # Phase 1: L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
    except Exception:
        pass

    if elapsed() > deadline:
        return np.clip(best_coeffs, -10, 10).tolist()

    # Phase 2: Nelder-Mead refinement
    try:
        res2 = minimize(objective, best_coeffs, method='Nelder-Mead',
                        options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-8, 'adaptive': True})
        if res2.fun < best_score:
            best_score = res2.fun
            best_coeffs = res2.x
    except Exception:
        pass

    if elapsed() > deadline:
        return np.clip(best_coeffs, -10, 10).tolist()

    # Phase 3: Polish with tighter tolerances
    try:
        res3 = minimize(objective, best_coeffs, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        if res3.fun < best_score:
            best_score = res3.fun
            best_coeffs = res3.x
    except Exception:
        pass

    # Final gamma constraint check
    for om_a, q, u, wa, x_vals, dx, D_ref, f_ref in precomputed:
        gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + \
                best_coeffs[3]*u + best_coeffs[4]*wa*a_vals + best_coeffs[5]*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            best_coeffs = x0.copy()
            break

    return np.clip(best_coeffs, -10, 10).tolist()
