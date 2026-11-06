"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def construct():
    """
    Return six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    """
    start_time = time.time()
    deadline = 82.0  # safety margin from 90s limit

    # Broader set of test cosmologies covering the likely validation space
    test_models = [
        (0.30, 0.0, 0.0),    # LCDM
        (0.30, -1.0, 0.0),   # Cosmological constant
        (0.30, -0.5, 0.5),   # CPL
        (0.30, -0.8, 0.3),   # CPL
        (0.30, -0.6, -0.2),  # CPL
        (0.30, -0.9, 0.2),   # More negative w0
        (0.30, -0.2, -0.4),  # Positive wa
        (0.25, -0.8, 0.3),   # Lower Om0
        (0.35, -0.5, 0.5),   # Higher Om0
        (0.25, -1.0, 0.0),   # Lower Om0, CC
        (0.35, -0.8, 0.3),   # Higher Om0, CPL
        (0.30, -0.7, 0.0),   # w0=-0.7, wa=0
        (0.30, -0.5, 0.0),   # w0=-0.5, wa=0
        (0.30, -1.0, 0.5),   # CC-like with positive wa
        (0.30, -0.8, -0.3),  # CPL with negative wa
    ]

    # Fine grid of a values
    a_values = np.linspace(0.1, 0.99, 50)

    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        return Om0 * a**-3 / E2

    def compute_reference_f_and_D(Om0, w0, wa, a_arr):
        """
        Compute reference f(a) and normalized D(a)/D(1) via RK4 on the
        GR second-order ODE for linear growth.
        """
        n_steps = 2000
        x_start = 0.0
        x_end = np.log(a_arr[0])
        h = (x_end - x_start) / n_steps

        def E2_func(a):
            return Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))

        def E2_prime(a):
            return -3*Om0*a**-4 + 3*wa*(1-Om0)*np.exp(3*wa*(1-a))

        def Hprime_over_H(a):
            return 0.5 * a * E2_prime(a) / E2_func(a)

        def Om_a_val(a):
            return Om0 * a**-3 / E2_func(a)

        def ode_rhs(x, y):
            a = np.exp(x)
            D, Dp = y
            HoH = Hprime_over_H(a)
            Om = Om_a_val(a)
            Dpp = -(3 + HoH) * Dp + 1.5 * Om * D
            return [Dp, Dpp]

        y = np.array([1.0, Om0 ** 0.55])
        x = x_start

        D_store = [1.0]
        f_store = [Om0 ** 0.55]
        x_store = [0.0]

        for i in range(n_steps):
            k1 = np.array(ode_rhs(x, y))
            k2 = np.array(ode_rhs(x + h/2, y + h/2 * k1))
            k3 = np.array(ode_rhs(x + h/2, y + h/2 * k2))
            k4 = np.array(ode_rhs(x + h, y + h * k3))
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            x = x + h
            D_store.append(y[0])
            f_store.append(y[1] / y[0])
            x_store.append(x)

        D_store = np.array(D_store)
        f_store = np.array(f_store)
        x_store = np.array(x_store)

        x_target = np.log(a_arr)
        D_ref = np.interp(x_target, x_store, D_store)
        f_ref = np.interp(x_target, x_store, f_store)
        D_norm = D_ref / D_ref[0]

        return f_ref, D_norm

    # Pre-compute references
    references = {}
    for Om0, w0, wa in test_models:
        f_ref, D_ref_norm = compute_reference_f_and_D(Om0, w0, wa, a_values)
        references[(Om0, w0, wa)] = (f_ref, D_ref_norm)

    def compute_error(coeffs):
        """Compute max relative error for both f and D across all models."""
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0

        for Om0, w0, wa in test_models:
            f_ref, D_ref_norm = references[(Om0, w0, wa)]

            # Compute approximate D via exp(-∫ f_approx dx)
            x_grid = np.log(a_values)
            f_approx_arr = np.zeros_like(a_values)
            gamma_arr = np.zeros_like(a_values)

            for i, a in enumerate(a_values):
                Om = Om_a(Om0, w0, wa, a)
                q = 1 - Om
                u = 1 + w0 + wa * (1 - a)

                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

                if gamma < 0.1 or gamma > 1.2:
                    return float('inf')

                f_approx_arr[i] = Om ** gamma
                gamma_arr[i] = gamma

            # Integrate f_approx from log(a) to 0 to get D_approx/D(1)
            # D(a)/D(1) = exp(-∫_log(a)^0 f(exp(x)) dx)
            # We have f at discrete x points; use trapezoidal rule
            x_sorted = np.sort(x_grid)
            f_sorted = f_approx_arr[np.argsort(x_grid)]
            # Integrate from x_sorted to 0
            # For each a_i, D_approx_i = exp(-∫_{x_i}^{0} f(x) dx)
            # Precompute cumulative integral from 0 downward
            # x_grid is from log(0.1) to log(0.99), i.e., negative to near 0
            # We want ∫_{x_i}^{0} f(x) dx
            # Use cumulative trapezoid
            dx = np.diff(x_sorted)
            trapz = 0.5 * (f_sorted[1:] + f_sorted[:-1]) * dx
            cumtrapz = np.concatenate([[0.0], np.cumsum(trapz)])
            # cumtrapz[i] = ∫_{x_sorted[0]}^{x_sorted[i]} f dx
            # We want ∫_{x_i}^{0} = ∫_{x_sorted[0]}^{0} - ∫_{x_sorted[0]}^{x_i}
            # ∫_{x_sorted[0]}^{0} ≈ cumtrapz[-1] + 0.5*(f_sorted[-1]+f(0))*(0-x_sorted[-1])
            # Simpler: just compute ∫_{x_i}^{0} by summing trapezoids from x_i to 0
            # Since x_sorted is increasing (from log(0.1)≈-2.3 to log(0.99)≈-0.01),
            # and 0 is just beyond the last point, approximate ∫_{x_i}^{0} ≈ 
            #   cumtrapz[-1] - cumtrapz[i] + 0.5*(f_sorted[-1]+f_sorted[i])*(0 - x_sorted[i])
            # Actually simpler: ∫_{x_i}^{0} ≈ 0.5*(f_sorted[i]+f_sorted[-1])*(0-x_sorted[i]) 
            # is a crude single-trapezoid; better to sum all trapezoids from i to end
            # plus a final half-step to 0.
            D_approx = np.zeros_like(a_values)
            for i in range(len(x_sorted)):
                # Sum trapezoids from index i to end of x_sorted
                if i < len(x_sorted) - 1:
                    integral_to_end = cumtrapz[-1] - cumtrapz[i]
                    # Add final segment from x_sorted[-1] to 0
                    final_seg = 0.5 * (f_sorted[-1] + f_sorted[i]) * (0 - x_sorted[i])
                    # Actually this isn't right either. Let me just do it properly.
                    integral = 0.0
                    for j in range(i, len(x_sorted)-1):
                        integral += 0.5*(f_sorted[j]+f_sorted[j+1])*(x_sorted[j+1]-x_sorted[j])
                    # Add segment from last point to 0
                    integral += 0.5*(f_sorted[-1]+f_sorted[i])*(0-x_sorted[i])
                else:
                    integral = 0.5*(f_sorted[i]+f_sorted[i])*(0-x_sorted[i])
                D_approx[i] = np.exp(-integral)

            # Map back to original ordering
            D_approx_original = np.zeros_like(a_values)
            sort_idx = np.argsort(x_grid)
            inv_sort = np.argsort(sort_idx)
            D_approx_original[inv_sort] = D_approx

            # Compute errors
            for i in range(len(a_values)):
                # f error
                if abs(f_ref[i]) > 1e-10:
                    err_f = abs(f_approx_arr[i] - f_ref[i]) / abs(f_ref[i])
                else:
                    err_f = abs(f_approx_arr[i] - f_ref[i])

                # D error
                if abs(D_ref_norm[i]) > 1e-10:
                    err_D = abs(D_approx_original[i] - D_ref_norm[i]) / abs(D_ref_norm[i])
                else:
                    err_D = abs(D_approx_original[i] - D_ref_norm[i])

                max_err = max(max_err, err_f, err_D)

        return max_err

    # Initial best coefficients from literature
    best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    best_score = compute_error(best_coeffs)

    # Global optimization with differential evolution
    if time.time() < deadline - 5:
        def objective(coeffs):
            if time.time() > deadline:
                return 1e6
            return compute_error(coeffs)

        bounds = [(-10, 10)] * 6

        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                update_interval=5,
                callback=None
            )
            if result.fun < best_score:
                best_coeffs = result.x
                best_score = result.fun
        except Exception:
            pass

    # Final validation: ensure gamma stays in [0.1, 1.2]
    valid = True
    for Om0, w0, wa in test_models:
        for a in a_values:
            Om = Om_a(Om0, w0, wa, a)
            q = 1 - Om
            u = 1 + w0 + wa * (1 - a)
            gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + \
                    best_coeffs[3]*u + best_coeffs[4]*wa*a + best_coeffs[5]*u*q
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break

    if not valid:
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    best_coeffs = np.clip(best_coeffs, -10, 10)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
