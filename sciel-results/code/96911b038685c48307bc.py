import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit

    Om0 = 0.3

    def get_e2(a, w0, wa):
        """Calculate E(a)^2 for CPL dark energy."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = a.reshape(1)
        x = 3.0 * wa * (1.0 - a)
        rho_de = a**(-3.0 * (1.0 + w0 + wa)) * np.exp(x)
        rho_m = a**(-3.0)
        e2 = Om0 * rho_m + (1.0 - Om0) * rho_de
        if scalar_input:
            return e2[0]
        return e2

    def get_om_m(a, w0, wa):
        """Calculate Omega_m(a)."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = a.reshape(1)
        e2_val = get_e2(a, w0, wa)
        rho_m = a**(-3.0)
        om_m = Om0 * rho_m / e2_val
        if scalar_input:
            return om_m[0]
        return om_m

    def linder_gamma(w0, wa):
        """Linder (2005) growth index approximation."""
        return 0.55 + 0.05 * (w0 - 0.5 * wa)

    def linder_f(a, w0, wa):
        """Linder approximation for growth rate."""
        om = get_om_m(a, w0, wa)
        g = linder_gamma(w0, wa)
        return om ** g

    def linder_D(a, w0, wa):
        """Linder approximation for normalized growth factor."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = a.reshape(1)
        D = np.ones_like(a)
        for i, av in enumerate(a):
            if av >= 1.0:
                D[i] = 1.0
                continue
            # Integrate f(a') da'/a' from av to 1
            n_pts = 200
            a_fine = np.linspace(av, 1.0, n_pts)
            f_fine = linder_f(a_fine, w0, wa)
            integral = np.trapz(f_fine / a_fine, a_fine)
            D[i] = np.exp(-integral)
        if scalar_input:
            return D[0]
        return D

    def approx_gamma(c, w0, wa, a):
        """Compute gamma from the 6-coefficient ansatz."""
        c0, c1, c2, c3, c4, c5 = c
        a = np.asarray(a, dtype=float)
        om_m = get_om_m(a, w0, wa)
        q = 1.0 - om_m
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        return gamma

    def approx_f(c, w0, wa, a):
        """Compute f from the 6-coefficient ansatz."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = a.reshape(1)
        gamma = approx_gamma(c, w0, wa, a)
        om_m = get_om_m(a, w0, wa)
        f = om_m ** gamma
        if scalar_input:
            return f[0]
        return f

    def approx_D(c, w0, wa, a):
        """Compute D from the 6-coefficient ansatz."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = a.reshape(1)
        D = np.ones_like(a)
        for i, av in enumerate(a):
            if av >= 1.0:
                D[i] = 1.0
                continue
            n_pts = 200
            a_fine = np.linspace(av, 1.0, n_pts)
            f_fine = approx_f(c, w0, wa, a_fine)
            integral = np.trapz(f_fine / a_fine, a_fine)
            D[i] = np.exp(-integral)
        if scalar_input:
            return D[0]
        return D

    # Build a dense training grid of cosmologies and scale factors
    w0_vals = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    wa_vals = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    a_vals = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

    # Collect training data: for each (w0, wa, a), compute Linder reference
    # We want to fit: gamma_ansatz(c, w0, wa, a) ≈ gamma_linder(w0, wa)
    # But since f = Om^gamma, matching gamma is the key.
    # Let's build a linear system: gamma_linder = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q

    train_rows = []
    train_targets = []

    for w0 in w0_vals:
        for wa in wa_vals:
            if time.time() - start_time > deadline - 20:
                break
            g_lind = linder_gamma(w0, wa)
            for av in a_vals:
                om_m = get_om_m(av, w0, wa)
                q = 1.0 - om_m
                u = 1.0 + w0 + wa * (1.0 - av)
                row = np.array([1.0, q, q**2, u, wa * av, u * q])
                train_rows.append(row)
                train_targets.append(g_lind)

    if len(train_rows) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(train_rows)
    b = np.array(train_targets)

    # Solve least squares: A @ c = b
    try:
        c_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        c_ls = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Clamp to [-10, 10]
    c_ls = np.clip(c_ls, -10.0, 10.0)

    # Check gamma constraint on training data
    def check_gamma_valid(c, w0_list, wa_list, a_list):
        for w0 in w0_list:
            for wa in wa_list:
                for av in a_list:
                    om_m = get_om_m(av, w0, wa)
                    q = 1.0 - om_m
                    u = 1.0 + w0 + wa * (1.0 - av)
                    g = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*av + c[5]*u*q
                    if g < 0.1 or g > 1.2:
                        return False
        return True

    # If least-squares solution violates gamma constraint, fall back to seed
    if not check_gamma_valid(c_ls, w0_vals[:5], wa_vals[:5], a_vals[:5]):
        c_ls = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Refinement: use nonlinear optimization to minimize max relative error
    # on a validation-like set of cosmologies
    val_w0 = np.array([-1.0, -0.5, 0.0, 0.5])
    val_wa = np.array([-0.5, 0.0, 0.5])
    val_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

    def objective(c):
        if time.time() - start_time > deadline - 5:
            return 1000.0
        c = np.clip(c, -10.0, 10.0)
        max_err = 0.0
        for w0 in val_w0:
            for wa in val_wa:
                f_ref = linder_f(val_a, w0, wa)
                f_app = approx_f(c, w0, wa, val_a)
                D_ref = linder_D(val_a, w0, wa)
                D_app = approx_D(c, w0, wa, val_a)

                # Check gamma constraint
                om_m = get_om_m(val_a, w0, wa)
                q = 1.0 - om_m
                u = 1.0 + w0 + wa * (1.0 - val_a)
                gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*val_a + c[5]*u*q
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    return 1000.0

                err_f = np.abs(f_app / f_ref - 1.0)
                err_D = np.abs(D_app / D_ref - 1.0)
                local_max = max(np.max(err_f), np.max(err_D))
                if local_max > max_err:
                    max_err = local_max
        return max_err * 100.0

    # Start refinement from least-squares solution
    best_c = c_ls.copy()
    best_val = objective(best_c)

    if time.time() - start_time < deadline - 15:
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_c = np.clip(res.x, -10.0, 10.0)
                best_val = res.fun
        except Exception:
            pass

    if time.time() - start_time < deadline - 5:
        try:
            res2 = minimize(objective, best_c, method='Powell',
                           options={'maxiter': 200, 'xrtol': 1e-7})
            if res2.fun < best_val:
                best_c = np.clip(res2.x, -10.0, 10.0)
                best_val = res2.fun
        except Exception:
            pass

    # Final validation: ensure gamma constraint holds on broader set
    final_w0 = np.array([-1.5, -1.2, -1.0, -0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0])
    final_wa = np.array([-0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8])
    final_a = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])

    gamma_valid = True
    for w0 in final_w0:
        for wa in final_wa:
            for av in final_a:
                om_m = get_om_m(av, w0, wa)
                q = 1.0 - om_m
                u = 1.0 + w0 + wa * (1.0 - av)
                g = best_c[0] + best_c[1]*q + best_c[2]*q**2 + best_c[3]*u + best_c[4]*wa*av + best_c[5]*u*q
                if g < 0.1 or g > 1.2:
                    gamma_valid = False
                    break
            if not gamma_valid:
                break
        if not gamma_valid:
            break

    if not gamma_valid:
        # Fall back to seed
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_c = np.clip(best_c, -10.0, 10.0)
    return list(best_c)

# EVOLVE-BLOCK-END
