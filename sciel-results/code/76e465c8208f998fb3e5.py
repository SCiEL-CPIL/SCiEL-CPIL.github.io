import numpy as np
import scipy.optimize
import scipy.integrate
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin for safety

    # Representative cosmologies for training
    # Omega_m0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    # We sample a grid to cover the likely validation space.
    om_m0_vals = [0.25, 0.3, 0.35]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.5, 0.0, 0.5]
    
    cosmologies = []
    for om in om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om, w0, wa))

    # Grid for a values to check errors
    # Log-spaced from 0.05 to 1.0
    a_grid = np.logspace(np.log10(0.05), 0.0, 100)
    x_grid = np.log(a_grid)

    # Pre-compute reference solutions
    # We need D_ref and f_ref for each cosmology on a_grid
    ref_data = []
    for (om_m0, w0, wa) in cosmologies:
        om_l0 = 1.0 - om_m0
        
        def w_of_a(a):
            return w0 + wa * (1.0 - a)
        
        def om_m_of_a(a):
            if a <= 0: return om_m0
            de_exponent = 3.0 * (1.0 + w0 + wa * (1.0 - a))
            # Note: The integral for DE density is rho_DE ~ a^{-3(1+w0+wa(1-a))}
            # Let's integrate exactly or use the standard formula.
            # rho_DE(a) = rho_DE(0) * exp(-3 * integral_0^a (1+w(a'))/a' da')
            # For CPL, this integral is 3(1+w0)ln(a) + 3wa(a-1)
            # So rho_DE(a) ~ a^{-3(1+w0)} * exp(-3wa(a-1))
            # Let's use the standard analytic form for CPL density.
            # E^2(a) = Om_m0 a^-3 + Om_l0 a^{-3(1+w0)} exp(-3wa(a-1))
            # Wait, the standard CPL density is:
            # rho_DE(a) = rho_DE(0) * a^{-3(1+w0)} * exp(-3*wa*(a-1))
            # Let's verify. d(rho)/da = -3(1+w) rho/a.
            # ln(rho) = -3(1+w0)ln(a) - 3wa(a-1) + C.
            # At a=1, ln(rho) = 0 (if normalized to 1).
            # So rho_DE(a) = a^{-3(1+w0)} * exp(-3wa(a-1)).
            
            de_term = om_l0 * (a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a - 1.0))
            m_term = om_m0 * (a ** -3.0)
            return m_term / (m_term + de_term)

        def H_of_x(x):
            a = np.exp(x)
            om_m = om_m_of_a(a)
            # H^2 = H0^2 E^2. We can work with E^2 or just the ratio.
            # The ODE is D'' + (3/2 + 3/2 w) D' - 3/2 Om_m D = 0
            # This equation is independent of H0.
            return om_m_of_a(a)

        # Integrate ODE from a=1 (x=0) backwards to a_min
        # D(1) = 1, D'(1) = f(1) = Om_m0^gamma_seed
        # We need a good seed for f(1). Constant gamma ~ 0.55 is standard.
        # Let's use a slightly better estimate for f(1) to ensure stability.
        # f(1) ~ Om_m0^0.55
        f_seed = om_m0 ** 0.55
        
        def ode_system(x, y):
            a = np.exp(x)
            om_m = om_m_of_a(a)
            w_a = w_of_a(a)
            # y[0] = D, y[1] = D'
            # D'' = - (1.5 + 1.5 w_a) D' + 1.5 om_m D
            dD = y[1]
            dDp = -1.5 * (1.0 + w_a) * y[1] + 1.5 * om_m * y[0]
            return [dD, dDp]

        # Integrate from 0 to -inf (or a small a)
        # We only need up to a=0.05
        x_min = np.log(0.05)
        # Use solve_ivp
        sol = scipy.integrate.solve_ivp(ode_system, [0, x_min], [1.0, f_seed], 
                                        method='RK45', rtol=1e-9, atol=1e-12,
                                        dense_output=True, max_step=0.1)
        
        # Evaluate on grid
        D_vals = np.ones_like(a_grid)
        f_vals = np.ones_like(a_grid)
        if sol.success:
            sol_grid = sol.sol(x_grid)
            D_vals = sol_grid[0]
            f_vals = sol_grid[1] / D_vals
        else:
            # Fallback to constant gamma if integration fails
            gamma_const = 0.55
            om_m_grid = np.array([om_m_of_a(a) for a in a_grid])
            f_vals = om_m_grid ** gamma_const
            # Integrate f to get D
            # D = exp(-int f dx)
            # Rough integration
            D_vals = np.exp(-np.cumsum(f_vals * np.diff(x_grid, prepend=0)))
            D_vals /= D_vals[-1] # Normalize to D(1)=1

        ref_data.append((a_grid, D_vals, f_vals, om_m0, w0, wa))

    # Objective function
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        sum_err_sq = 0.0
        n_errs = 0
        
        for (a_grid, D_ref, f_ref, om_m0, w0, wa) in ref_data:
            om_m_grid = np.array([om_m0 * (a**-3) / (om_m0 * (a**-3) + (1-om_m0) * (a**(-3*(1+w0))) * np.exp(-3*wa*(a-1))) for a in a_grid])
            q_grid = 1.0 - om_m_grid
            u_grid = 1.0 + w0 + wa * (1.0 - a_grid)
            
            # Calculate gamma
            gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
            
            # Check gamma bounds
            if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
                return 1e10 # Penalty
            
            # Calculate f_approx
            f_approx = om_m_grid ** gamma_grid
            
            # Calculate D_approx
            # D(a)/D(1) = exp(-int_0^ln(a) f(e^x) dx)
            # We need to integrate f_approx from 0 to x
            # Use cumulative trapezoidal rule on x_grid
            # x_grid is log(a_grid). a_grid is increasing, so x_grid is increasing.
            # But we integrate from 0 to x. Since x_grid starts at log(0.05) < 0, we need to be careful.
            # The formula is D(a)/D(1) = exp(- \int_{\ln a}^0 f(e^x) dx) = exp(\int_0^{\ln a} ... wait.
            # D(a) = D(1) exp(- \int_0^a f(a')/a' da')
            # Let x = ln a. dx = da/a.
            # D(a)/D(1) = exp(- \int_0^x f(e^t) dt)
            
            # We have f_approx on a_grid (which corresponds to x_grid).
            # x_grid is from log(0.05) to 0.
            # We need integral from 0 to x.
            # Let's reverse the grid for integration from 0 downwards.
            x_rev = -x_grid[::-1] # 0 to -log(0.05)
            f_rev = f_approx[::-1]
            
            # Cumulative integral from 0
            # np.cumsum(f_rev * dx)
            dx = x_rev[1] - x_rev[0]
            # Trapezoidal
            # I[0] = 0
            # I[i] = I[i-1] + 0.5*(f[i-1]+f[i])*dx
            integrals = np.zeros_like(x_grid)
            integrals_rev = np.zeros_like(x_rev)
            for i in range(1, len(x_rev)):
                integrals_rev[i] = integrals_rev[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * dx
            integrals = integrals_rev[::-1]
            
            D_approx = np.exp(-integrals)
            
            # Relative errors
            err_f = np.abs((f_approx - f_ref) / f_ref)
            err_D = np.abs((D_approx - D_ref) / D_ref)
            
            # Max error for this cosmology
            max_err_c = np.max(np.concatenate([err_f, err_D]))
            if max_err_c > max_err:
                max_err = max_err_c
            
            # For smooth optimization, we can use a high power sum
            # But max is non-smooth. Let's use a soft max.
            # Actually, for the final answer, we just want to minimize the max.
            # For the optimizer, a sum of squares of errors might be better to find a good basin,
            # then we can polish with max.
            # However, the score is based on MAX error.
            # Let's return a combination: 0.1 * max_err + 0.9 * mean(err^2)
            # Or just mean(err^2) to guide the gradient, then check max at the end.
            # Given the small dimension (6), we can use Nelder-Mead on the max error directly if we make it slightly smooth.
            # Let's use the sum of errors squared as the primary objective for the optimizer,
            # as it is differentiable and guides towards low error regions.
            # We will verify the max error at the end.
            
            err_vec = np.concatenate([err_f, err_D])
            sum_err_sq += np.sum(err_vec**2)
            n_errs += len(err_vec)
        
        if n_errs == 0:
            return 1e10
            
        mean_sq_err = sum_err_sq / n_errs
        # Penalize if max_err is high
        # The task score is 100 * max_relative_error.
        # We want to minimize this.
        # Let's return a smooth approximation of the max.
        # max(x) ~ (1/k) * sum(x^k) for large k.
        # But we don't have all errors in one array easily.
        # Let's just return mean_sq_err. The optimizer will find a point where errors are small everywhere.
        # Then we can do a final polish on the max.
        
        return mean_sq_err

    # Initial guess
    # Standard constant gamma is 0.55.
    # Let's start with [0.55, 0, 0, 0, 0, 0]
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0
    best_obj = objective(x0)
    
    # Optimization
    # Use L-BFGS-B with bounds
    bounds = [(-10, 10)] * 6
    
    # We will try a few restarts with different initial guesses if time permits
    # But first, one solid optimization run.
    
    try:
        result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < best_obj:
            best_coeffs = result.x
            best_obj = result.fun
            
        # Polish with Nelder-Mead on the max error metric?
        # Let's define a max error objective
        def max_error_objective(coeffs):
            c0, c1, c2, c3, c4, c5 = coeffs
            max_err = 0.0
            for (a_grid, D_ref, f_ref, om_m0, w0, wa) in ref_data:
                om_m_grid = np.array([om_m0 * (a**-3) / (om_m0 * (a**-3) + (1-om_m0) * (a**(-3*(1+w0))) * np.exp(-3*wa*(a-1))) for a in a_grid])
                q_grid = 1.0 - om_m_grid
                u_grid = 1.0 + w0 + wa * (1.0 - a_grid)
                gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
                if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
                    return 1e10
                f_approx = om_m_grid ** gamma_grid
                
                x_rev = -x_grid[::-1]
                f_rev = f_approx[::-1]
                dx = x_rev[1] - x_rev[0]
                integrals_rev = np.zeros_like(x_rev)
                for i in range(1, len(x_rev)):
                    integrals_rev[i] = integrals_rev[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * dx
                integrals = integrals_rev[::-1]
                D_approx = np.exp(-integrals)
                
                err_f = np.abs((f_approx - f_ref) / f_ref)
                err_D = np.abs((D_approx - D_ref) / D_ref)
                max_err_c = np.max(np.concatenate([err_f, err_D]))
                if max_err_c > max_err:
                    max_err = max_err_c
            return max_err

        # Check time
        if time.time() < deadline - 10.0:
            # Polish with Nelder-Mead on max error
            result2 = scipy.optimize.minimize(max_error_objective, best_coeffs, method='Nelder-Mead',
                                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result2.fun < max_error_objective(best_coeffs):
                best_coeffs = result2.x
                
    except Exception as e:
        pass # Keep best_coeffs as is

    # Final validation and clipping
    c = best_coeffs
    # Ensure coefficients are within [-10, 10]
    c = np.clip(c, -10, 10)
    
    # Check gamma constraints one last time on a dense grid
    # If invalid, revert to safe constant
    safe_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid = True
    test_a = np.logspace(np.log10(0.01), 0.0, 200)
    for (om_m0, w0, wa) in cosmologies:
        om_m_grid = np.array([om_m0 * (a**-3) / (om_m0 * (a**-3) + (1-om_m0) * (a**(-3*(1+w0))) * np.exp(-3*wa*(a-1))) for a in test_a])
        q_grid = 1.0 - om_m_grid
        u_grid = 1.0 + w0 + wa * (1.0 - test_a)
        gamma_grid = c[0] + c[1]*q_grid + c[2]*q_grid**2 + c[3]*u_grid + c[4]*wa*test_a + c[5]*u_grid*q_grid
        if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
            valid = False
            break
            
    if not valid:
        c = safe_coeffs
        
    return c.tolist()
# EVOLVE-BLOCK-END
