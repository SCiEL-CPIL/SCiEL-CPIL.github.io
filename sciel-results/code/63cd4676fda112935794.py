"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Helper functions
    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        # E(a)^2 = Om0/a^3 + (1-Om0) * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        # For CPL: w(a) = w0 + wa*(1-a)
        # rho_de(a) ~ a^(-3(1+w)) where w varies
        # Using the standard CPL form:
        # E^2(a) = Om_m/a^3 + Om_de * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        Om_de = 1.0 - Om0
        # More careful: for CPL, the DE density evolves as
        # rho_de(a) = rho_de(1) * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        # So E^2(a) = Om_m/a^3 + Om_de * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
        """Compute reference f and D/D(1) using GR ODE."""
        n = len(a_grid)
        # Solve the growth equation: D'' + (2+a'/a)/H * D' - 3/2 * Om_m(a) * D = 0
        # In terms of x = ln(a), dD/dx = D', d2D/dx2 = D''
        # H^2 = H0^2 * E^2(a)
        # a'/a = H = H0*E(a) (in units where H0=1)
        # So: D'' + 2*E(a)*D' - 3/2 * Om_m(a) * D = 0
        
        def ode(x, y):
            a = np.exp(x)
            E2 = Om0/a**3 + (1-Om0)*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
            E = np.sqrt(E2)
            Om_m = Om0/a**3 / E2
            # D'' = -2*E*D' + 3/2*Om_m*D
            dD = y[1]
            d2D = -2.0*E*dD + 1.5*Om_m*y[0]
            return [dD, d2D]
        
        # Integrate from x=ln(a_min) to x=0 (a=1)
        # Use D(1)=1, D'(1) = f(1) = Om0^gamma ~ 0.55 as initial guess
        # Actually integrate forward from early times
        x_min = np.log(a_grid[0])
        x_max = 0.0
        
        # Integrate from x_min to x_max with initial conditions at x_min
        # D(x_min) ~ a_min (matter era), D'(x_min) ~ 1
        a_min = a_grid[0]
        y_init = [a_min, 1.0]
        
        sol = solve_ivp(ode, [x_min, x_max], y_init, 
                       t_eval=a_grid if False else np.linspace(x_min, x_max, 200),
                       method='RK45', rtol=1e-10, atol=1e-12)
        
        if not sol.success:
            return None, None
        
        # Interpolate to get values at a_grid
        x_grid = np.log(a_grid)
        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])
        
        # Normalize: D/D(1)
        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        
        # f = D'/D
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
        # Clip gamma to [0.1, 1.2]
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
        """Compute normalized D from the model via quadrature."""
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # = exp(-integral from a to 1 of f(a')/a' da')
        # Use numerical quadrature
        n = len(a)
        D_norm = np.zeros(n)
        D_norm[-1] = 1.0  # D(1)/D(1) = 1
        
        # Integrate from a=1 backwards
        # For each a_i, compute integral from a_i to 1
        # Use cumulative trapezoidal rule
        f_vals = model_f(c, a, Om0, w0, wa)
        # integral from a_i to 1 of f(a')/a' da'
        # Change variable: x = ln(a), dx = da/a
        # integral from x_i to 0 of f(exp(x)) dx
        x = np.log(a)
        # We need integral from x_i to 0
        # Use cumulative integration from the end
        # dx = x[i] - x[i-1] for forward, but we go backward
        # Let's compute cumulative from right
        dx = np.diff(x)  # positive since x is increasing
        # trapezoidal: integral from x_i to x_{i+1} of f dx ≈ 0.5*(f_i + f_{i+1})*dx_i
        trap = 0.5*(f_vals[:-1] + f_vals[1:]) * dx
        # Cumulative sum from the end: integral from x_i to 0
        # integral from x_i to 0 = sum of trap from i to n-2
        cum = np.zeros(n)
        cum[-2] = trap[-1]
        for i in range(n-3, -1, -1):
            cum[i] = cum[i+1] + trap[i]
        # D(a_i)/D(1) = exp(-cum[i])
        D_norm[:-1] = np.exp(-cum)
        return D_norm

    def objective(c, cosmologies, a_grid):
        """Compute max relative error over all cosmologies and a values."""
        max_err = 0.0
        for (Om0, w0, wa) in cosmologies:
            f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
            if f_ref is None:
                continue
            f_mod = model_f(c, a_grid, Om0, w0, wa)
            D_mod = model_D(c, a_grid, Om0, w0, wa)
            
            # Relative errors
            err_f = np.abs((f_mod - f_ref) / f_ref)
            err_D = np.abs((D_mod - D_ref) / D_ref)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        return max_err

    # Define validation cosmologies (typical CPL models)
    cosmologies = [
        (0.3, -0.5, 0.0),    # LCDM
        (0.3, -1.0, 0.0),    # Cosmological constant
        (0.3, -0.9, 0.2),    # CPL with positive wa
        (0.3, -0.9, -0.2),   # CPL with negative wa
        (0.25, -0.8, 0.1),   # Lower matter, moderate wa
        (0.35, -1.0, 0.0),   # Higher matter
    ]
    
    a_grid = np.linspace(0.05, 1.0, 50)
    
    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)
    
    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Fast local optimization using Nelder-Mead
    from scipy.optimize import minimize
    
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
    
    # Start from constant gamma
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Try Nelder-Mead with time check
    best_c = c0.copy()
    best_err = fast_objective(best_c)
    
    if time.time() > DEADLINE - 10:
        return best_c.tolist()
    
    result = minimize(fast_objective, best_c, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.success or result.fun < best_err:
        best_c = result.x
        best_err = result.fun
    
    if time.time() > DEADLINE - 5:
        return best_c.tolist()
    
    # Try a few more local optimizations from different starting points
    starts = [
        np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, 0.05, 0.0, 0.0]),
    ]
    
    for s in starts:
        if time.time() > DEADLINE - 3:
            break
        res = minimize(fast_objective, s, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_err:
            best_c = res.x
            best_err = res.fun
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return best_c.tolist()
# EVOLVE-BLOCK-END
