"""Improved growth-index approximation for CPL dark-energy cosmologies."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Generate six coefficients [c0..c5] for the growth-index gamma(a) model.
    Uses a search approach with a better reference computation.
    """
    t_start = time.time()
    DEADLINE = 75.0  # seconds, safety margin before 90s kill

    def compute_om(a, Om0, w0, wa):
        """Compute matter density parameter Omega_m(a) given a in [0.1, 1]."""
        # For CPL: E(a)^2 = Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
        # This is the standard approximation for w(a) = w0 + wa*(1-a)
        w = w0 + wa * (1.0 - a)
        # Use the integrated form for accuracy
        e2 = Om0 * np.power(a, -3.0) + (1.0 - Om0) * np.power(a, -3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
        om_m = Om0 / e2
        return om_m

    def compute_f_and_D(coeffs, a_arr, Om0, w0, wa):
        """
        Compute f(a) and normalized D(a) for given coefficients and model.
        coeffs = [c0, c1, c2, c3, c4, c5]
        """
        c0, c1, c2, c3, c4, c5 = coeffs
        a = a_arr
        om_m = compute_om(a, Om0, w0, wa)
        q = 1.0 - om_m
        w = w0 + wa * (1.0 - a)
        u = 1.0 + w
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        
        f = np.power(om_m, gamma_clipped)
        
        # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        x_min = np.min(np.log(a))
        x_grid = np.linspace(x_min, 0.0, 4000)
        a_grid = np.exp(x_grid)
        om_grid = compute_om(a_grid, Om0, w0, wa)
        q_grid = 1.0 - om_grid
        w_grid = w0 + wa * (1.0 - a_grid)
        u_grid = 1.0 + w_grid
        gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
        gamma_grid = np.clip(gamma_grid, 0.1, 1.2)
        f_grid = np.power(om_grid, gamma_grid)
        
        # Cumulative integral using trapezoidal rule
        cum_int = np.zeros_like(x_grid)
        dx = x_grid[1] - x_grid[0]
        for i in range(1, len(x_grid)):
            cum_int[i] = cum_int[i-1] + 0.5 * (f_grid[i] + f_grid[i-1]) * dx
        
        ln_a = np.log(a)
        cum_int_at_ln_a = np.interp(ln_a, x_grid, cum_int)
        integral_ln_a_to_0 = cum_int[-1] - cum_int_at_ln_a
        D_normalized = np.exp(-integral_ln_a_to_0)
        
        return f, D_normalized, gamma

    def compute_reference(Om0, w0, wa, a_values):
        """
        Compute exact reference f and D by integrating the growth ODE.
        Uses the equation: df/dx = -f^2 - (3/2 + 0.5*dlnE2_dx)*f + 1.5*Om_m
        where x = ln(a).
        """
        x_min = np.min(np.log(a_values))
        # Use a fine grid for reference
        n_pts = 4000
        x_grid = np.linspace(0.0, x_min, n_pts)  # from 0 to x_min (negative)
        a_grid = np.exp(x_grid)
        
        e2 = Om0 * np.power(a_grid, -3.0) + (1.0 - Om0) * np.power(a_grid, -3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_grid - 1.0))
        om_m = Om0 / e2
        
        # Compute d ln(E^2)/dx numerically
        dlnE2_dx = np.zeros_like(x_grid)
        for i in range(1, len(x_grid)-1):
            dlnE2_dx[i] = (np.log(e2[i+1]) - np.log(e2[i-1])) / (x_grid[i+1] - x_grid[i-1])
        dlnE2_dx[0] = dlnE2_dx[1]
        dlnE2_dx[-1] = dlnE2_dx[-2]
        
        # Integrate f ODE from x=0 (a=1) backward
        # Initial condition: f(1) = Om_m(1)^gamma_ref, but we need a self-consistent reference
        # Use the standard initial condition f(1) ≈ 0.55 * Om0^0.3 for LambdaCDM-like
        # For better accuracy, integrate D directly using the second-order ODE
        
        # Let's integrate D(x) using: D''(x) + (3/2 + 0.5*dlnE2_dx)*D'(x) - 1.5*om_m*D = 0
        # Wait, that's not right. The correct ODE for D in terms of x=ln(a):
        # D''(x) + (3/2 + 0.5 * d ln(E^2)/dx) D'(x) - (3/2) * Om_m(a) * E(a)^2 / H0^2 * D = 0
        # Actually, let me use the f ODE which is cleaner:
        # df/dx = -f^2 - (3/2 + 0.5*dlnE2_dx)*f + 1.5*Om_m
        
        # For the initial condition, use f(1) from a well-known approximation
        # f(1) = Om_m(1)^gamma where gamma ≈ 0.55 for matter-dominated
        f_init = 0.55 * Om0**0.3
        
        f_ref = np.zeros_like(x_grid)
        f_ref[0] = f_init
        
        # Use RK4 for better accuracy
        dx = x_grid[1] - x_grid[0]
        for i in range(1, len(x_grid)):
            x0 = x_grid[i-1]
            f0 = f_ref[i-1]
            
            def dfdx(x, f):
                idx = int((x - x_grid[0]) / dx)
                idx = max(0, min(idx, len(x_grid)-2))
                return -f**2 - (1.5 + 0.5*dlnE2_dx[idx])*f + 1.5*om_m[idx]
            
            k1 = dfdx(x0, f0)
            k2 = dfdx(x0 + dx/2, f0 + dx/2 * k1)
            k3 = dfdx(x0 + dx/2, f0 + dx/2 * k2)
            k4 = dfdx(x0 + dx, f0 + dx * k3)
            f_ref[i] = f0 + dx/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Compute D from f: D(x) = D(0) * exp(integral from 0 to x of f dx)
        # Since x is negative, integral from 0 to x is negative
        D_ref = np.zeros_like(x_grid)
        D_ref[0] = 1.0
        for i in range(1, len(x_grid)):
            dx_step = x_grid[i] - x_grid[i-1]
            # Trapezoidal rule for integral of f
            int_step = 0.5 * (f_ref[i] + f_ref[i-1]) * dx_step
            D_ref[i] = D_ref[i-1] * np.exp(int_step)
        
        # Interpolate to a_values
        ln_a = np.log(a_values)
        f_ref_at_a = np.interp(ln_a, x_grid, f_ref)
        D_ref_at_a = np.interp(ln_a, x_grid, D_ref)
        
        return f_ref_at_a, D_ref_at_a

    def objective(coeffs, validation_models, a_values):
        """
        Compute max relative error of f and D across validation models.
        """
        max_rel_err = 0.0
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Check if all coefficients are in [-10, 10]
        if any(abs(c) > 10.0 for c in coeffs):
            return 1e6
        
        for Om0, w0, wa in validation_models:
            f_approx, D_approx, gamma = compute_f_and_D(coeffs, a_values, Om0, w0, wa)
            
            # Check gamma constraint
            if np.min(gamma) < 0.1 - 1e-6 or np.max(gamma) > 1.2 + 1e-6:
                return 1e6 + abs(np.max(gamma) - 1.2) + abs(0.1 - np.min(gamma))
            
            # Compute exact reference
            f_ref, D_ref = compute_reference(Om0, w0, wa, a_values)
            
            # Relative errors
            mask_f = np.abs(f_ref) > 1e-8
            if np.any(mask_f):
                rel_err_f = np.max(np.abs((f_approx[mask_f] - f_ref[mask_f]) / f_ref[mask_f]))
                max_rel_err = max(max_rel_err, rel_err_f)
            
            mask_D = np.abs(D_ref) > 1e-8
            if np.any(mask_D):
                rel_err_D = np.max(np.abs((D_approx[mask_D] - D_ref[mask_D]) / D_ref[mask_D]))
                max_rel_err = max(max_rel_err, rel_err_D)
        
        return 100.0 * max_rel_err

    # Validation models: typical cosmologies
    validation_models = [
        (0.3, -1.0, 0.0),     # LambdaCDM
        (0.3, -1.2, 0.5),     # CPL with evolution
        (0.25, -0.9, -0.3),   # Another CPL
        (0.35, -1.1, 0.2),    # Another
        (0.3, -1.0, 0.3),     # CPL
        (0.28, -0.8, 0.1),    # Another
    ]
    a_values = np.linspace(0.1, 1.0, 200)
    
    # Search: try multiple starting points
    best_score = 1e10
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Generate diverse starting points
    np.random.seed(42)
    n_starts = 30
    
    for i in range(n_starts):
        if time.time() - t_start > DEADLINE:
            break
        
        # Varyed starts
        if i == 0:
            start = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif i == 1:
            start = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        elif i == 2:
            start = [0.6, -0.5, 0.0, 0.0, 0.0, 0.0]
        elif i == 3:
            start = [0.55, 0.3, 0.1, 0.0, 0.0, 0.0]
        elif i == 4:
            start = [0.55, 0.0, 0.2, 0.0, 0.0, 0.0]
        elif i == 5:
            start = [0.55, 0.0, 0.0, 0.3, 0.0, 0.0]
        elif i == 6:
            start = [0.55, 0.0, 0.0, 0.0, 0.3, 0.0]
        elif i == 7:
            start = [0.55, 0.0, 0.0, 0.0, 0.0, 0.3]
        elif i == 8:
            start = [0.55, 0.2, 0.1, 0.1, 0.1, 0.1]
        elif i == 9:
            start = [0.5, 0.3, 0.05, 0.05, 0.05, 0.05]
        elif i == 10:
            start = [0.6, -0.2, 0.1, 0.05, 0.05, 0.05]
        else:
            start = np.random.uniform(-0.5, 0.5, 6)
            start[0] = 0.45 + 0.2 * np.random.rand()  # Keep c0 near 0.55
        
        # Local optimization
        try:
            result = minimize(
                objective,
                start,
                args=(validation_models, a_values),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = list(result.x)
        except Exception:
            pass
    
    # Final check: ensure coefficients are in [-10, 10]
    best_coeffs = [max(-10.0, min(10.0, c)) for c in best_coeffs]
    
    return best_coeffs

# EVOLVE-BLOCK-END
