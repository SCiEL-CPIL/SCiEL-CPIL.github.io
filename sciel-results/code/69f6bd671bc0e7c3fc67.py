"""Cosmology: compact dark-energy growth approximation - search-based constructor."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # Plausible validation cosmology grid (CPL dark energy)
    # Om0 in [0.15, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    # a values from ~0.05 to 1.0
    om0_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    a_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Build evaluation grid
    grid = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    grid.append((om0, w0, wa, a))
    
    def Om_of_a(om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        # EdS-like background: H^2(a)/H0^2 = om0*a^-3 + (1-om0)*(a/(1+a*wa))^(3*(1+w0+wa))
        # For CPL: de density ∝ a^(-3(1+w0+wa)) = a^(-3-3w0-3wa)
        # But with wa, the effective exponent changes with a
        # Omega_de(a) = (1-om0) * a^(-3*(1+w0+wa*(1-a)))... 
        # Actually for CPL: w(a) = w0 + wa*(1-a)
        # rho_de(a) = rho_de(1) * exp(-3*int_1^a (1+w(a'))/a' da')
        # = rho_de(1) * a^(-3(1+w0)) * exp(-3*wa*int_1^a (1-a')/a' da')
        # int_1^a (1-a')/a' da' = int_1^a (1/a' - 1) da' = ln(a) - (a-1)
        # So rho_de(a) ∝ a^(-3(1+w0)) * exp(-3*wa*(ln(a) - a + 1))
        # = a^(-3(1+w0)) * exp(-3*wa*ln(a)) * exp(3*wa*(a-1))
        # = a^(-3(1+w0) - 3*wa) * exp(3*wa*(a-1))
        # = a^(-3-3w0-3wa) * exp(3*wa*(a-1))
        
        # Let's use a simpler form: for the background
        # E^2(a) = om0/a^3 + (1-om0) * a^(-3*(1+w0+wa*(1-a)))  -- this is approximate
        # More carefully:
        # H^2(a)/H0^2 = om0 * a^-3 + (1-om0) * a^(-3*(1+w0)) * exp(-3*wa*(ln(a)-a+1))
        
        if a <= 0:
            return om0
        w_eff = w0 + wa * (1 - a)
        # Use the exact CPL integration
        # int_1^a (1+w(a'))/a' da' = int_1^a (1+w0+wa-wa*a')/a' da'
        # = (1+w0+wa)*ln(a) - wa*(a-1)
        # So rho_de(a)/rho_de(1) = exp(-3*[(1+w0+wa)*ln(a) - wa*(a-1)])
        # = a^(-3*(1+w0+wa)) * exp(3*wa*(a-1))
        de_factor = a**(-3.0*(1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
        E2 = om0 * a**(-3.0) + (1.0 - om0) * de_factor
        Om = om0 * a**(-3.0) / E2
        return Om
    
    def gamma_from_coeffs(coeffs, om0, w0, wa, a):
        """Compute gamma from the 6-coefficient formula."""
        c0, c1, c2, c3, c4, c5 = coeffs
        Om = Om_of_a(om0, w0, wa, a)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        return gamma, Om, q, u
    
    def f_approx(coeffs, om0, w0, wa, a):
        """Approximate growth rate f = Om^gamma."""
        gamma, Om, q, u = gamma_from_coeffs(coeffs, om0, w0, wa, a)
        # Clip gamma to [0.1, 1.2]
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om**gamma_clipped, gamma
    
    def true_f(om0, w0, wa, a):
        """True growth rate from the GR ODE (approximated via the known relation).
        
        The true growth rate f = d ln D / d ln a satisfies:
        f' + f^2 + (1 + 0.5 q) f = 0.5 Om  (in appropriate variables)
        
        For a good approximation, we can use the fact that for EdS, f = Om^(5/7).
        For general CPL, we can solve the ODE numerically.
        """
        # Solve the growth ODE: f' + f^2 + (1 + 0.5*q)*f = 0.5*Om
        # where q = 1 - Om
        # We integrate from a=1 backwards, with f(1) ≈ Om(1)^(5/7) as initial condition
        
        # Actually, let's use a simpler approach: the true f can be computed
        # by solving the ODE. But for speed, let's use a high-accuracy approximation.
        # The most common high-accuracy approximation is f = Om^gamma_true
        # where gamma_true ≈ 0.55 + 0.05*w0 + 0.01*wa (roughly)
        # But we need the TRUE value. Let's solve the ODE.
        
        # For efficiency, use a fixed-step RK4 from a=1 to a
        if a >= 1.0:
            return Om_of_a(om0, w0, wa, 1.0)**(5.0/7.0)
        
        # Integrate ODE from a=1 to a (backwards)
        # f' = -f^2 - (1 + 0.5*q)*f + 0.5*Om
        # where q = 1 - Om(a)
        
        n_steps = 200
        a_grid = np.linspace(1.0, a, n_steps + 1)
        da = a_grid[1] - a_grid[0]
        
        f_val = Om_of_a(om0, w0, wa, 1.0)**(5.0/7.0)
        
        for i in range(n_steps):
            a_curr = a_grid[i]
            Om_curr = Om_of_a(om0, w0, wa, a_curr)
            q_curr = 1.0 - Om_curr
            
            # RK4
            def dfda(f, a_pt):
                Om_pt = Om_of_a(om0, w0, wa, a_pt)
                q_pt = 1.0 - Om_pt
                return -f**2 - (1.0 + 0.5*q_pt)*f + 0.5*Om_pt
            
            k1 = dfda(f_val, a_curr)
            k2 = dfda(f_val + 0.5*da*k1, a_curr + 0.5*da)
            k3 = dfda(f_val + 0.5*da*k2, a_curr + 0.5*da)
            k4 = dfda(f_val + da*k3, a_curr + da)
            f_val = f_val + (da/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        return f_val
    
    def D_normalized_approx(coeffs, om0, w0, wa, a):
        """Compute D(a)/D(1) = exp(-int_0^a f(exp(x)) dx) where x = ln(a)."""
        if a >= 1.0:
            return 1.0
        
        # Integrate f(exp(x)) from x=ln(a) to x=0
        # i.e., int_a^1 f(a')/a' da'
        # Let x = ln(a'), dx = da'/a'
        # So int_{ln(a)}^0 f(e^x) dx
        
        x_start = np.log(a)
        x_end = 0.0
        n_steps = 100
        x_grid = np.linspace(x_start, x_end, n_steps + 1)
        dx = x_grid[1] - x_grid[0]
        
        integral = 0.0
        for i in range(n_steps):
            x_mid = 0.5 * (x_grid[i] + x_grid[i+1])
            a_mid = np.exp(x_mid)
            f_mid, _ = f_approx(coeffs, om0, w0, wa, a_mid)
            integral += f_mid * dx
        
        return np.exp(-integral)
    
    def D_normalized_true(om0, w0, wa, a):
        """True normalized growth factor."""
        if a >= 1.0:
            return 1.0
        
        # Integrate the true f
        x_start = np.log(a)
        x_end = 0.0
        n_steps = 100
        x_grid = np.linspace(x_start, x_end, n_steps + 1)
        dx = x_grid[1] - x_grid[0]
        
        integral = 0.0
        for i in range(n_steps):
            x_mid = 0.5 * (x_grid[i] + x_grid[i+1])
            a_mid = np.exp(x_mid)
            f_mid = true_f(om0, w0, wa, a_mid)
            integral += f_mid * dx
        
        return np.exp(-integral)
    
    def objective(coeffs):
        """Compute worst-case relative error across f and D."""
        max_err = 0.0
        for (om0, w0, wa, a) in grid:
            f_app, gamma = f_approx(coeffs, om0, w0, wa, a)
            f_tru = true_f(om0, w0, wa, a)
            
            # Check gamma constraint
            if gamma < 0.1 or gamma > 1.2:
                return 1e6  # penalty
            
            if abs(f_tru) > 1e-10:
                err_f = abs(f_app - f_tru) / abs(f_tru)
            else:
                err_f = abs(f_app - f_tru)
            
            D_app = D_normalized_approx(coeffs, om0, w0, wa, a)
            D_tru = D_normalized_true(om0, w0, wa, a)
            
            if abs(D_tru) > 1e-10:
                err_D = abs(D_app - D_tru) / abs(D_tru)
            else:
                err_D = abs(D_app - D_tru)
            
            max_err = max(max_err, err_f, err_D)
        
        return 100.0 * max_err
    
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-10, 10)] * 6
    
    best_coeffs = x0
    best_score = objective(x0)
    
    # Try differential evolution for global search
    if time.time() - t0 < DEADLINE - 20:
        try:
            result = differential_evolution(
                objective, bounds, 
                maxiter=50, popsize=15, 
                tol=1e-4, seed=42,
                polish=False
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Polish with L-BFGS-B
    if time.time() - t0 < DEADLINE - 5:
        try:
            result = minimize(
                objective, best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish
    if time.time() - t0 < DEADLINE - 2:
        try:
            result = minimize(
                objective, best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
