"""Growth-index optimization via hardcoded optimal coefficients with fast local polish."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
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

        # Use a moderately fine grid for speed
        x_fine = np.linspace(x_min, x_max, 400)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-10, atol=1e-12)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        f_ref = dD_ref / D_ref

        return f_ref, D_ref

    def model_gamma(c, a, Om0, w0, wa):
        q = 1.0 - Om_of_a(a, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def model_f(c, a, Om0, w0, wa):
        gamma = model_gamma(c, a, Om0, w0, wa)
        Om_m = Om_of_a(a, Om0, w0, wa)
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
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
    a_grid = np.linspace(0.05, 1.0, 100)

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

    # Hardcoded initial guess based on known good approximations for CPL models
    # This is a structural change: we start from a very good point, not a random one.
    # Typical gamma is around 0.5-0.6.
    best_c = np.array([0.55, 0.1, -0.1, 0.05, 0.0, -0.05])
    
    # Ensure initial guess is valid
    for (Om0, w0, wa) in cosmologies:
        g = model_gamma(best_c, a_grid, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
            break

    best_err = fast_objective(best_c)

    # Use a fast local optimizer (L-BFGS-B) to refine. This is much faster than DE.
    # We use bounds to keep coefficients in a reasonable range to avoid numerical issues.
    bounds = [(-10, 10)] * 6
    
    # Tighter bounds for better convergence
    bounds = [(0.1, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    try:
        result = minimize(
            fast_objective,
            best_c,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success and result.fun < best_err:
            best_c = result.x
            best_err = result.fun
    except Exception:
        pass

    if time.time() > DEADLINE - 5:
        return np.clip(best_c, -10, 10).tolist()

    # Final polish: try a few random perturbations around best_c
    np.random.seed(123)
    for _ in range(10):
        if time.time() > DEADLINE - 3:
            break
        perturbation = np.random.normal(0, 0.02, 6)
        c_trial = best_c + perturbation
        c_trial = np.clip(c_trial, -10, 10)
        
        # Check constraints
        valid = True
        for (Om0, w0, wa) in cosmologies:
            g = model_gamma(c_trial, a_grid, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                valid = False
                break
        
        if valid:
            err = fast_objective(c_trial)
            if err < best_err:
                best_err = err
                best_c = c_trial.copy()

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
