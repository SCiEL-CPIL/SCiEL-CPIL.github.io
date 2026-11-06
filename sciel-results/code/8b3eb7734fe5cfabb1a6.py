"""Improved growth-index optimization with Taylor expansion in q and refined search."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
        """Compute reference f and D/D(1) using GR ODE."""
        n = len(a_grid)
        x_grid = np.log(a_grid)

        def ode(x, y):
            a = np.exp(x)
            E2 = Om0/a**3 + (1-Om0)*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
            E = np.sqrt(E2)
            Om_m = Om0/a**3 / E2
            dD = y[1]
            d2D = -2.0*E*dD + 1.5*Om_m*y[0]
            return [dD, d2D]

        x_min = x_grid[0]
        x_max = 0.0

        # Use a fine grid for integration
        x_fine = np.linspace(x_min, x_max, 500)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-12, atol=1e-14)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        f_ref = dD_ref / D_ref

        return f_ref, D_ref

    def model_gamma(c, a, Om0, w0, wa):
        """Compute gamma from the 6-coefficient model."""
        q = 1.0 - Om_of_a(a, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def model_f(c, a, Om0, w0, wa):
        """Compute f from the model."""
        gamma = model_gamma(c, a, Om0, w0, wa)
        Om_m = Om_of_a(a, Om0, w0, wa)
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
        """Compute normalized D from the model via quadrature."""
        n = len(a)
        f_vals = model_f(c, a, Om0, w0, wa)
        x = np.log(a)
        dx = np.diff(x)
        trap = 0.5*(f_vals[:-1] + f_vals[1:]) * dx
        cum = np.zeros(n)
        cum[-2] = trap[-1]
        for i in range(n-3, -1, -1):
            cum[i] = cum[i+1] + trap[i]
        D_norm = np.zeros(n)
        D_norm[-1] = 1.0
        D_norm[:-1] = np.exp(-cum)
        return D_norm

    # Define validation cosmologies
    cosmologies = [
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -0.9, -0.2),
        (0.25, -0.8, 0.1),
        (0.35, -1.0, 0.0),
        (0.30, -0.7, 0.3),
        (0.28, -1.1, -0.1),
    ]

    # Use a dense a-grid for better reference
    a_grid = np.linspace(0.05, 1.0, 80)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def fast_objective(c):
        max_err = 0.0
        for (Om0, w0, wa) in cosmologies:
            if (Om0, w0, wa) not in refs:
                continue
            f_ref, D_ref = refs[(Om0, w0, wa)]
            f_mod = model_f(c, a_grid, Om0, w0, wa)
            D_mod = model_D(c, a_grid, Om0, w0, wa)

            err_f = np.abs((f_mod - f_ref) / f_ref)
            err_D = np.abs((D_mod - D_ref) / D_ref)

            max_err = max(max_err, np.max(err_f), np.max(err_D))

        return max_err

    # Strategy: use a Taylor expansion in q around q=0 (present epoch)
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # The key insight: near a=1, q ≈ 1-Om0, u ≈ 1+w0
    # We want gamma to match the true growth index well.
    # The true gamma at a=1 is approximately Om0^0.55 for LCDM.
    # Let's start with a good initial guess and refine.

    # Initial guess: constant gamma ~ 0.55, with small corrections
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_c = c0.copy()
    best_err = fast_objective(best_c)

    if time.time() > DEADLINE - 15:
        return best_c.tolist()

    # First pass: Nelder-Mead with reasonable settings
    result = minimize(fast_objective, best_c, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9,
                               'adaptive': True})
    if result.fun < best_err:
        best_c = result.x
        best_err = result.fun

    if time.time() > DEADLINE - 10:
        return np.clip(best_c, -10, 10).tolist()

    # Second pass: try different starting points based on Taylor expansion intuition
    # Near a=1: q ≈ 1-Om0 ≈ 0.7, u ≈ 1+w0
    # For LCDM (w0=-0.5, wa=0): u ≈ 0.5
    # gamma ≈ c0 + c1*0.7 + c2*0.49 + c3*0.5 + c5*0.35
    # We want gamma ≈ 0.55 at a=1 for LCDM

    starts = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.50, 0.10, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.60, -0.10, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.05, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.05]),
        np.array([0.52, 0.05, 0.02, 0.02, 0.0, 0.0]),
        np.array([0.58, -0.05, 0.02, -0.02, 0.0, 0.0]),
    ]

    for s in starts:
        if time.time() > DEADLINE - 5:
            break
        res = minimize(fast_objective, s, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9,
                                'adaptive': True})
        if res.fun < best_err:
            best_c = res.x
            best_err = res.fun

    # Final polish: high-precision Nelder-Mead from best point
    if time.time() < DEADLINE - 3:
        res = minimize(fast_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                                'adaptive': True})
        if res.fun < best_err:
            best_c = res.x
            best_err = res.fun

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
