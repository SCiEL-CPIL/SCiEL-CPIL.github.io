"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    # Optimize six coefficients for the growth-index formula
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s limit
    
    # Define plausible validation cosmologies
    # (Om0, w0, wa) - covering LambdaCDM and CPL variants
    cosmologies = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.1),      # CPL with positive wa
        (0.3, -1.0, -0.1),     # CPL with negative wa
        (0.3, -0.9, 0.0),      # w0 slightly above -1
        (0.3, -1.1, 0.0),      # w0 slightly below -1
        (0.3, -0.9, 0.1),
        (0.3, -0.9, -0.1),
        (0.3, -1.1, 0.1),
        (0.3, -1.1, -0.1),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
    ]
    
    # Scale factors to evaluate
    a_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    def Om_of_a(a, Om0, w0, wa):
        """Compute Om(a) for CPL dark energy."""
        de_factor = (1 + w0 + wa * (1 - a))
        if abs(de_factor) < 1e-10:
            # Degenerate case, handle carefully
            return Om0 * a**(-3) / (Om0 * a**(-3) + 1e-10)
        de_power = 3 * (1 + de_factor)
        de_term = (1 - Om0) * a**(-de_power)
        matter_term = Om0 * a**(-3)
        return matter_term / (matter_term + de_term)
    
    def true_f(a, Om0, w0, wa):
        """True growth rate from ODE."""
        Om_a = Om_of_a(a, Om0, w0, wa)
        # Solve the ODE for D(a)
        # D'' + (3/a + H'/H)D' - (3/2)Om(a)/a^2 D = 0
        # In terms of x = ln(a), with d/dx = a d/da
        # D_xx + (3/2 + H_x/H) D_x - (3/2)Om(a) D = 0
        # where H_x = dH/dx = H * (3/2 * Om(a) + 3/2 * Ode(a) * (1+w(a)))
        
        # Use a simpler approach: solve the second-order ODE numerically
        # For the growth rate f = d ln D / d ln a = D' / D (where ' is d/d ln a)
        
        # Actually, let's use the standard approach: integrate the ODE for D
        # and compute f = d ln D / d ln a
        
        # Define the ODE in terms of x = ln(a)
        # Let y1 = D, y2 = dD/dx
        # y2' = - (3/2 + E_x) * y2 + (3/2) * Om(exp(x)) * y1
        # where E_x = d ln E / dx, E = H/H0
        
        # For CPL: E^2 = Om0 * exp(-3x) + (1-Om0) * exp(-3*(1+w0+wa*(1-exp(x))))
        # This is complex. Let's use a different approach.
        
        # Simpler: use the fact that for a given cosmology, we can compute
        # the growth rate using the approximation or solve the ODE.
        
        # Let's solve the ODE numerically from a=0.1 to a=1
        # and extract f at the desired a.
        
        if a <= 0:
            return 0.55
        
        x_target = np.log(a)
        
        def ode_system(x, y):
            a_x = np.exp(x)
            Om_a = Om_of_a(a_x, Om0, w0, wa)
            Ode_a = 1 - Om_a
            w_a = w0 + wa * (1 - a_x)
            
            # H^2 = H0^2 * [Om_m * a^-3 + Om_de * a^(-3(1+w))]
            # dH/dln(a) = H * (3/2 * Om_m + 3/2 * Ode * (1+w))
            # So E_x = d ln E / d ln a = (3/2) * Om_m + (3/2) * Ode * (1+w)
            E_x = 1.5 * Om_a + 1.5 * Ode_a * (1 + w_a)
            
            y1, y2 = y
            dy1 = y2
            dy2 = -(1.5 + E_x) * y2 + 1.5 * Om_a * y1
            return [dy1, dy2]
        
        # Initial conditions at small a (matter dominated)
        x_start = np.log(0.05)
        # In matter dominated era, D ~ a^(6/5), so f ~ 6/5
        y_start = [0.05**(6/5), (6/5) * 0.05**(6/5)]
        
        # Integrate to x_target
        x_eval = np.linspace(x_start, x_target, 100)
        sol = solve_ivp(ode_system, [x_start, x_target], y_start, 
                       t_eval=x_eval, rtol=1e-8, atol=1e-10, method='RK45')
        
        if not sol.success:
            return 0.55
        
        # Find the value at x_target
        idx = np.argmin(np.abs(sol.t - x_target))
        D_val = sol.y[0, idx]
        dD_dx = sol.y[1, idx]
        
        if abs(D_val) < 1e-10:
            return 0.55
            
        f_val = dD_dx / D_val
        return f_val
    
    def compute_D_normalized(a, Om0, w0, wa):
        """Compute D(a)/D(1) using numerical integration."""
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # where f is the true growth rate
        
        # We need to compute the integral of f(x) from x=ln(a) to x=0
        # Let's compute f at many points and integrate
        
        if a >= 1.0:
            return 1.0
            
        x_start = np.log(a)
        x_end = 0.0
        
        # Evaluate f at multiple points
        n_points = 50
        x_vals = np.linspace(x_start, x_end, n_points)
        f_vals = []
        for x in x_vals:
            a_x = np.exp(x)
            f_vals.append(true_f(a_x, Om0, w0, wa))
        f_vals = np.array(f_vals)
        
        # Integrate using trapezoidal rule
        integral = np.trapz(f_vals, x_vals)
        
        return np.exp(-integral)
    
    def model_gamma(c, a, Om0, w0, wa):
        """Compute gamma from coefficients."""
        Om_a = Om_of_a(a, Om0, w0, wa)
        q = 1 - Om_a
        u = 1 + w0 + wa * (1 - a)
        c0, c1, c2, c3, c4, c5 = c
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        return gamma
    
    def model_f(c, a, Om0, w0, wa):
        """Compute model f."""
        Om_a = Om_of_a(a, Om0, w0, wa)
        gamma = model_gamma(c, a, Om0, w0, wa)
        if Om_a <= 0:
            return 0.0
        return Om_a ** gamma
    
    def model_D_normalized(c, a, Om0, w0, wa):
        """Compute model D(a)/D(1)."""
        if a >= 1.0:
            return 1.0
        
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # Use model f for the integral
        x_start = np.log(a)
        x_end = 0.0
        n_points = 50
        x_vals = np.linspace(x_start, x_end, n_points)
        f_vals = []
        for x in x_vals:
            a_x = np.exp(x)
            f_vals.append(model_f(c, a_x, Om0, w0, wa))
        f_vals = np.array(f_vals)
        integral = np.trapz(f_vals, x_vals)
        return np.exp(-integral)
    
    def objective(c):
        """Compute max relative error across all cosmologies and a values."""
        max_err = 0.0
        for (Om0, w0, wa) in cosmologies:
            for a in a_vals:
                # True values
                f_true = true_f(a, Om0, w0, wa)
                D_true = compute_D_normalized(a, Om0, w0, wa)
                
                # Model values
                f_model = model_f(c, a, Om0, w0, wa)
                D_model = model_D_normalized(c, a, Om0, w0, wa)
                
                # Relative errors
                if abs(f_true) > 1e-10:
                    err_f = abs(f_model - f_true) / abs(f_true)
                else:
                    err_f = abs(f_model - f_true)
                
                if abs(D_true) > 1e-10:
                    err_D = abs(D_model - D_true) / abs(D_true)
                else:
                    err_D = abs(D_model - D_true)
                
                max_err = max(max_err, err_f, err_D)
        
        return max_err
    
    # Initial guess: constant gamma = 0.55
    c0_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds: coefficients in [-10, 10]
    bounds = [(-10, 10)] * 6
    
    # Try optimization
    best_c = c0_init
    best_obj = objective(c0_init)
    
    # Nelder-Mead optimization
    try:
        result = minimize(objective, c0_init, method='Nelder-Mead', 
                         bounds=bounds, 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        if time.time() < deadline:
            if result.fun < best_obj:
                best_c = result.x
                best_obj = result.fun
    except Exception:
        pass
    
    # Try a few more local optimizations from different starting points
    if time.time() < deadline:
        start_points = [
            [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.60, 0.1, 0.0, 0.0, 0.0, 0.0],
        ]
        for sp in start_points:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(objective, sp, method='Nelder-Mead',
                                 bounds=bounds,
                                 options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                          'adaptive': True})
                if result.fun < best_obj:
                    best_c = result.x
                    best_obj = result.fun
            except Exception:
                pass
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    
    # Convert to list of floats
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END

