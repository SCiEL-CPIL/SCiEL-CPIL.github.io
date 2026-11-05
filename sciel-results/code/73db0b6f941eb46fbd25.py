"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize, least_squares
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = 1e10

    def om_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a_safe = np.maximum(a, 1e-10)
        E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1+w0)) * np.exp(3*wa*(a_safe - 1))
        return Om0 * a_safe**(-3) / E2

    def solve_ode_fast(a_vals, Om0, w0, wa):
        """Solve growth ODE for multiple a values efficiently."""
        if len(a_vals) == 0:
            return np.array([]), np.array([])
        
        a_min = max(min(a_vals), 1e-6)
        a_max = max(a_vals)
        
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
        
        sol = solve_ivp(ode, [x_start, x_end], y0, rtol=1e-8, atol=1e-10, 
                       t_eval=np.log(a_vals), dense_output=True)
        
        if sol.success:
            f_vals = sol.y[0]
            D_vals = sol.y[1]
            # Normalize D to D(1)=1
            D_vals = D_vals / D_vals[-1]
            return f_vals, D_vals
        else:
            return None, None

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
        """Approximate normalized D for vector of a values."""
        if len(a_vals) == 0:
            return np.array([])
        
        # Integrate f from ln(a) to 0
        a_sorted = np.sort(a_vals)
        a_sorted = np.maximum(a_sorted, 1e-6)
        
        # Use quadrature on log scale
        x_vals = np.linspace(np.log(a_sorted[0]), 0, max(200, len(a_sorted)*10))
        a_interp = np.exp(x_vals)
        f_interp = f_approx_vec(c, a_interp, Om0, w0, wa)
        
        # Integrate -f dx from x to 0
        integral = np.zeros(len(a_sorted))
        for i, a_i in enumerate(a_sorted):
            x_i = np.log(a_i)
            mask = x_vals >= x_i
            if np.sum(mask) > 1:
                integral[i] = np.trapz(f_interp[mask], x_vals[mask])
            else:
                integral[i] = 0.0
        
        D_vals = np.exp(-integral)
        
        # Map back to original order
        sort_idx = np.argsort(a_vals)
        D_original = np.empty_like(D_vals)
        D_original[sort_idx] = D_vals
        return D_original

    def compute_log_f_deriv_residuals(c, a_fine, models):
        """
        Compute residuals for log-derivative fitting.
        For f = Om^gamma, ln(f) = gamma * ln(Om), so d(ln f)/dx = gamma * d(ln Om)/dx + (d gamma/dx) * ln(Om)
        We want to match d(ln f)/dx from the ODE solution.
        """
        residuals = []
        for Om0, w0, wa in models:
            # Get exact f values on fine grid
            f_exact, _ = solve_ode_fast(a_fine, Om0, w0, wa)
            if f_exact is None:
                continue
            
            om = om_a(a_fine, Om0, w0, wa)
            x = np.log(a_fine)
            
            # Compute d(ln f)/dx from exact solution using central differences
            ln_f = np.log(np.maximum(f_exact, 1e-10))
            dlnf_dx = np.gradient(ln_f, x)
            
            # Compute d(ln Om)/dx analytically: d(ln Om)/dx = -3*Om + 3*E^2 where E^2 = Om + Omega_de
            E2 = 1.0 / om
            dlnOm_dx = -3.0 * om + 3.0 * E2
            
            # For our approximation: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
            # d gamma / dx = (dg/dq)*(dq/dx) + (dg/du)*(du/dx) + c4*wa*da/dx
            # q = 1 - Om, dq/dx = -dOm/dx = 3*Om - 3*E2
            # u = 1 + w0 + wa*(1-a), du/dx = -wa*da/dx = -wa*a
            # da/dx = a
            
            q = 1 - om
            u = 1 + w0 + wa*(1 - a_fine)
            
            dq_dx = 3.0 * om - 3.0 * E2  # = -dlnOm_dx
            du_dx = -wa * a_fine
            
            dg_dq = c[1] + 2*c[2]*q + c[5]*u
            dg_du = c[3] + c[5]*q
            
            dgamma_dx = dg_dq * dq_dx + dg_du * du_dx + c[4]*wa*a_fine
            
            ln_Om = np.log(np.maximum(om, 1e-10))
            gamma_approx = gamma_vec(c, a_fine, Om0, w0, wa)
            
            # Approximate d(ln f)/dx = gamma * dlnOm_dx + dgamma_dx * ln_Om
            dlnf_approx = gamma_approx * dlnOm_dx + dgamma_dx * ln_Om
            
            # Residual: difference in d(ln f)/dx, weighted
            res = dlnf_approx - dlnf_dx
            residuals.append(res)
        
        if not residuals:
            return np.array([])
        
        return np.concatenate(residuals)

    def objective(c):
        """Compute 100 * max relative error with smooth soft-max."""
        if time.time() > deadline - 5:
            return 1e10
        
        max_err = 0.0
        err_terms = []
        
        for Om0, w0, wa in test_models:
            f_exact, D_exact = solve_ode_fast(a_values, Om0, w0, wa)
            if f_exact is None:
                continue
            
            f_app = f_approx_vec(c, a_values, Om0, w0, wa)
            D_app = D_approx_vec(c, a_values, Om0, w0, wa)
            
            # Check gamma bounds
            g = gamma_vec(c, a_values, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                return 1e10
            
            # Relative errors
            err_f = np.abs(f_app - f_exact) / np.maximum(np.abs(f_exact), 1e-10)
            err_D = np.abs(D_app - D_exact) / np.maximum(np.abs(D_exact), 1e-10)
            
            err_terms.extend(err_f.tolist())
            err_terms.extend(err_D.tolist())
        
        if not err_terms:
            return 1e10
        
        err_arr = np.array(err_terms)
        # Soft-max for smoothness
        beta = 20.0
        log_max = np.log(np.sum(np.exp(beta * err_arr))) / beta
        return 100.0 * log_max

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
    
    # Fine grid for derivative fitting
    a_fine = np.linspace(0.1, 1.0, 50)

    # Initial guess
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10, 10)] * 6

    try:
        # Step 1: Use least-squares on log-derivative residuals to get a good initial guess
        # This provides a more stable optimization landscape
        if time.time() < deadline - 30:
            lsq_result = least_squares(
                compute_log_f_deriv_residuals,
                c0,
                args=(a_fine, test_models),
                bounds=([-10]*6, [10]*6),
                method='trf',
                max_nfev=200,
                xtol=1e-8,
                ftol=1e-8,
            )
            
            if lsq_result.success and lsq_result.fun < 1e10:
                # Check gamma bounds for the LSQ solution
                c_lsq = lsq_result.x
                g_check = gamma_vec(c_lsq, a_fine, 0.3, -1.0, 0.0)
                if np.all(g_check >= 0.1) and np.all(g_check <= 1.2):
                    c0 = c_lsq
                    print(f"LSQ initial guess: {c0}, cost: {lsq_result.fun:.6f}")
        
        # Step 2: Use differential evolution for global search from improved initial guess
        de_result = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=80, 
            tol=1e-6,
            popsize=12,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            updating='immediate',
            x0=c0,
        )
        
        if time.time() < deadline - 20:
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_c = de_result.x
        
        # Step 3: Polish with L-BFGS-B
        if time.time() < deadline - 10:
            result = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
            
            if time.time() < deadline - 5 and result.success:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_c = result.x
        
        # Step 4: Final polish with Nelder-Mead if time remains
        if time.time() < deadline - 5:
            result2 = minimize(objective, best_c, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if time.time() < deadline - 2 and result2.success:
                score2 = result2.fun
                if score2 < best_score:
                    best_score = score2
                    best_c = result2.x
        
        best_c = np.clip(best_c, -10, 10)
        return list(best_c)
        
    except Exception as e:
        print(f"Exception in construct: {e}")
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END
