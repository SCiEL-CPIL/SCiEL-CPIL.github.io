"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 82  # safety margin before 90s kill
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = 1e10

    def om_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a_safe = np.maximum(np.asarray(a, dtype=float), 1e-10)
        E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1+w0)) * np.exp(3*wa*(a_safe - 1))
        return Om0 * a_safe**(-3) / E2

    def precompute_exact(Om0, w0, wa, a_eval):
        """Solve the growth ODE once for a given cosmology and return f and D at a_eval."""
        a_min = max(min(a_eval), 1e-6)
        a_max = max(a_eval)

        def ode(x, y):
            a_x = np.exp(x)
            om_x = om_a(a_x, Om0, w0, wa)
            f_x = y[0]
            D_x = y[1]
            dfdx = -f_x**2 - 1.5*(1 - om_x)*f_x + 1.5
            dDdx = f_x * D_x
            return [dfdx, dDdx]

        x_start = np.log(a_min)
        x_end = 0.0
        om_1 = om_a(1.0, Om0, w0, wa)
        y0 = [om_1**0.55, 1.0]

        # Use dense grid for interpolation
        x_grid = np.linspace(x_start, x_end, 2000)
        sol = solve_ivp(ode, [x_start, x_end], y0, rtol=1e-10, atol=1e-12,
                       t_eval=x_grid)

        if not sol.success:
            return None, None

        f_grid = sol.y[0]
        D_grid = sol.y[1]
        D_grid = D_grid / D_grid[-1]  # Normalize D(1)=1

        # Interpolate to a_eval
        x_eval = np.log(np.maximum(a_eval, 1e-6))
        # Sort x_grid and x_eval for interpolation
        f_eval = np.interp(x_eval, sol.t, f_grid)
        D_eval = np.interp(x_eval, sol.t, D_grid)

        return f_eval, D_eval

    def gamma_vec(c, a_vals, Om0, w0, wa):
        """Compute gamma for vector of a values."""
        om = om_a(a_vals, Om0, w0, wa)
        q = 1 - om
        u = 1 + w0 + wa*(1 - a_vals)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q

    def f_approx_vec(c, a_vals, Om0, w0, wa):
        """Approximate f for vector of a values."""
        g = gamma_vec(c, a_vals, Om0, w0, wa)
        om = om_a(a_vals, Om0, w0, wa)
        return np.power(om, g)

    def D_approx_vec(c, a_vals, Om0, w0, wa):
        """Approximate normalized D for vector of a values using cumulative trapezoid."""
        if len(a_vals) == 0:
            return np.array([])

        a_sorted = np.sort(a_vals)
        a_sorted = np.maximum(a_sorted, 1e-6)

        # Create fine grid for integration
        x_fine = np.linspace(np.log(a_sorted[0]), 0, 2000)
        a_fine = np.exp(x_fine)
        f_fine = f_approx_vec(c, a_fine, Om0, w0, wa)

        # Cumulative integral from x to 0: integral = total - cumsum_up_to_x
        # Use trapezoidal rule
        dx = np.diff(x_fine)
        avg_f = 0.5 * (f_fine[:-1] + f_fine[1:])
        trap_areas = dx * avg_f
        cum_areas = np.concatenate([[0.0], np.cumsum(trap_areas)])
        total_area = cum_areas[-1]

        # For each a_i, integral from ln(a_i) to 0 = total - integral from start to ln(a_i)
        x_eval = np.log(a_sorted)
        # Interpolate cum_areas at x_eval
        integral_vals = total_area - np.interp(x_eval, x_fine, cum_areas)

        D_vals = np.exp(-integral_vals)

        # Map back to original order
        sort_idx = np.argsort(a_vals)
        D_original = np.empty_like(D_vals)
        D_original[sort_idx] = D_vals
        return D_original

    # Validation cosmologies
    test_models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -0.5, -0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.2, 0.4),
        (0.30, -0.8, -0.3),
    ]
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Pre-compute exact references
    exact_data = []
    for Om0, w0, wa in test_models:
        f_exact, D_exact = precompute_exact(Om0, w0, wa, a_values)
        if f_exact is not None:
            exact_data.append((Om0, w0, wa, f_exact, D_exact))

    def objective(c):
        """Compute 100 * max relative error (direct max, not soft-max)."""
        if time.time() > deadline - 3:
            return 1e10

        max_err = 0.0

        for Om0, w0, wa, f_exact, D_exact in exact_data:
            f_app = f_approx_vec(c, a_values, Om0, w0, wa)
            D_app = D_approx_vec(c, a_values, Om0, w0, wa)

            # Check gamma bounds
            g = gamma_vec(c, a_values, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                return 1e10

            # Relative errors
            err_f = np.abs(f_app - f_exact) / np.maximum(np.abs(f_exact), 1e-10)
            err_D = np.abs(D_app - D_exact) / np.maximum(np.abs(D_exact), 1e-10)

            max_err = max(max_err, np.max(err_f), np.max(err_D))

        return 100.0 * max_err

    # Initial guess
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10, 10)] * 6

    try:
        # Phase 1: Differential Evolution for global exploration
        if time.time() < deadline - 30:
            result = differential_evolution(
                objective, bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                update_interval=10,
                callback=lambda xk, convergence: convergence
            )
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_c = result.x

        # Phase 2: L-BFGS-B polish from best DE result
        if time.time() < deadline - 15:
            result2 = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if result2.success and result2.fun < best_score:
                best_score = result2.fun
                best_c = result2.x

        # Phase 3: Nelder-Mead final polish
        if time.time() < deadline - 8:
            result3 = minimize(objective, best_c, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result3.success and result3.fun < best_score:
                best_score = result3.fun
                best_c = result3.x

        best_c = np.clip(best_c, -10, 10)
        return list(best_c)

    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END
