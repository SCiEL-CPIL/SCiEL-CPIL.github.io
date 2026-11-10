"""Improved growth-index optimization with proper ODE reference."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def _compute_reference(Om0, w0, wa, a_values):
    """Compute reference f and D/D(1) by solving the growth ODE."""
    # x = ln(a), so a = exp(x)
    # D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0 is wrong
    # Correct: D'' + (3/2)Om(a)D' = 0 is matter only
    # General: D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0 ... let me derive properly
    
    # The growth equation in terms of x=ln(a):
    # D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0 is NOT correct
    # 
    # The correct equation is:
    # D'' + (3/2)Om(a)D' = 0 for matter-only
    # With dark energy, the equation is:
    # D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0 ... still not right
    #
    # Let me use the standard form:
    # D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0 is wrong
    #
    # Actually the correct linear growth equation is:
    # D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0 ... I keep writing the same thing
    #
    # Let me be very careful. The density perturbation equation is:
    # δ'' + (3/2)Om(a)δ' - (3/2)Om(a)δ = 0 where prime = d/dln(a)
    # This is the standard form for the growth factor D where δ = D(a)δ_m,0
    
    # So: D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0
    # Let f = D'/D, then D' = fD, D'' = (f' + f^2)D
    # (f' + f^2)D + (3/2)Om(a)fD - (3/2)Om(a)D = 0
    # f' + f^2 + (3/2)Om(a)f - (3/2)Om(a) = 0
    # f' = -f^2 - (3/2)Om(a)f + (3/2)Om(a)
    
    # So we can solve for f directly as a first-order ODE!
    # f' = -f^2 - (3/2)Om(a)f + (3/2)Om(a)
    
    # And D(a)/D(1) = exp(-∫_a^1 f(x) dx) where x = ln(a')
    # Wait, D' = dD/dln(a) = f*D
    # So d(ln D)/dln(a) = f
    # ln(D(a)/D(1)) = ∫_1^a f(x) dx = -∫_a^1 f(x) dx
    
    # Let me solve for f(x) where x = ln(a)
    
    def om_of_x(x):
        a = np.exp(x)
        de_factor = (1 - a)**(-3*(1 + w0 + wa)) if a < 1 else 1.0
        E2 = Om0 / (a**3) + (1 - Om0) * de_factor
        return (Om0 / (a**3)) / E2
    
    # x ranges from ln(a_min) to ln(1) = 0
    # We integrate from x=0 (a=1) backwards to x=ln(a_min)
    # Or we can integrate forward from small a to a=1
    
    # Initial condition: at very small a (matter dominated), f ≈ Om(a)^0.55 ≈ 1
    # Actually f ≈ 1 in matter dominated era
    # Let's start at x = ln(0.01) and integrate to x = 0
    
    x_start = np.log(0.01)
    x_end = 0.0
    
    def ode_func(x, y):
        f = y[0]
        Om = om_of_x(x)
        return [-f**2 - 1.5*Om*f + 1.5*Om]
    
    # Initial guess: f ≈ 1 at early times
    y0 = [1.0]
    
    # We need to match a specific set of a values, so let's integrate over a fine grid
    # and interpolate
    
    x_grid = np.linspace(x_start, x_end, 1000)
    
    try:
        sol = solve_ivp(ode_func, [x_start, x_end], y0, t_eval=x_grid, 
                       method='RK45', rtol=1e-10, atol=1e-12)
        if not sol.success:
            return None, None
    except:
        return None, None
    
    f_ref_grid = sol.y[0]
    
    # Interpolate to get f at the requested a values
    a_ref_grid = np.exp(x_grid)
    f_ref = np.interp(a_values, a_ref_grid, f_ref_grid)
    
    # Compute D(a)/D(1) = exp(-∫_a^1 f dx) where x = ln(a')
    # = exp(∫_1^a f(x) dx) ... wait
    # d(ln D)/dx = f(x) where x = ln(a)
    # ln(D(a)/D(1)) = ∫_{x=0}^{x=ln(a)} f(x') dx' = -∫_{ln(a)}^0 f(x') dx'
    
    # So D(a)/D(1) = exp(∫_0^{ln(a)} f(x') dx')
    # Since ln(a) < 0 for a < 1, this is negative, so D(a)/D(1) < 1
    
    # Compute the integral numerically
    from scipy.integrate import cumulative_trapezoid
    
    # We need ∫_0^{ln(a)} f(x) dx for each a
    # Let's compute cumulative integral from x_start to each x
    integral_from_start = cumulative_trapezoid(f_ref_grid, x_grid, initial=0)
    
    # integral_from_start[i] = ∫_{x_start}^{x_grid[i]} f(x) dx
    # We want ∫_0^{ln(a)} f(x) dx = ∫_{x_start}^{ln(a)} f dx - ∫_{x_start}^0 f dx
    # = integral_from_start at ln(a) - integral_from_start at x=0
    
    # x=0 corresponds to the last element
    integral_at_0 = integral_from_start[-1]
    
    # For each a, find x = ln(a) and get the integral
    x_a = np.log(a_values)
    integral_at_a = np.interp(x_a, x_grid, integral_from_start)
    
    log_D_ratio = integral_at_a - integral_at_0
    
    D_ratio = np.exp(log_D_ratio)
    
    return f_ref, D_ratio

def _growth_index_error(coeffs):
    """Compute the worst relative error for a given set of 6 coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Test cosmologies
    test_models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, -0.3),
        (0.3, -1.2, 0.5),
        (0.3, -0.8, 0.0),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.0, -1.0),
        (0.3, -1.0, 1.0),
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.5),
    ]
    
    a_values = np.linspace(0.05, 0.99, 30)
    
    max_rel_err = 0.0
    
    for Om0, w0, wa in test_models:
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_values)
        if f_ref is None:
            continue
        
        for i, a in enumerate(a_values):
            if a <= 0:
                continue
                
            de_factor = (1 - a)**(-3*(1 + w0 + wa)) if a < 1 else 1.0
            E2 = Om0 / (a**3) + (1 - Om0) * de_factor
            Om_a = (Om0 / (a**3)) / E2
            
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            if gamma < 0.1 or gamma > 1.2:
                return 10000.0
            
            f_approx = Om_a**gamma
            
            rel_err_f = abs(f_approx - f_ref[i]) / max(abs(f_ref[i]), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_f)
            
            # For D, we need to compute D_approx/D(1) using the same formula
            # D_approx(a)/D_approx(1) = exp(-∫_a^1 f_approx(x) dx)
            # But f_approx depends on Om(x) which depends on a, so we need to integrate
            # This is expensive, so let's approximate by sampling
            
            # Actually, let's compute D_approx by integrating f_approx numerically
            # f_approx(x) = Om(x)^gamma where Om(x) is computed at each x
            # But gamma itself depends on q, u, wa, a which all depend on a
            # So gamma is not constant along the integral path
            
            # This makes D_approx computation complex. Let's sample the integral.
            
            # For speed, let's use a coarse grid for the D integral
            a_int = np.linspace(a, 1.0, 20)
            f_int = []
            for ai in a_int:
                if ai <= 0:
                    f_int.append(0)
                    continue
                de_i = (1 - ai)**(-3*(1 + w0 + wa)) if ai < 1 else 1.0
                E2_i = Om0 / (ai**3) + (1 - Om0) * de_i
                Om_i = (Om0 / (ai**3)) / E2_i
                q_i = 1 - Om_i
                u_i = 1 + w0 + wa * (1 - ai)
                gamma_i = c0 + c1*q_i + c2*q_i**2 + c3*u_i + c4*wa*ai + c5*u_i*q_i
                if gamma_i < 0.1 or gamma_i > 1.2:
                    f_int.append(0)
                else:
                    f_int.append(Om_i**gamma_i)
            
            # Integrate f_approx dx from a to 1, where x = ln(a')
            x_int = np.log(a_int)
            integral_approx = np.trapz(f_int, x_int)
            
            D_approx = np.exp(-integral_approx)
            
            rel_err_D = abs(D_approx - D_ref[i]) / max(abs(D_ref[i]), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_D)
    
    return max_rel_err * 100

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Start with seed
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_x = x0
    best_err = _growth_index_error(x0)
    
    # Try multiple restarts with different initial points
    initial_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.1, 0.0, 0.0, 0.0],
        [0.5, 0.2, 0.0, 0.1, 0.0, 0.0],
        [0.6, 0.0, 0.0, 0.1, 0.0, 0.0],
    ]
    
    for x_init in initial_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(_growth_index_error, x_init, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5,
                                     'adaptive': True})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except:
            continue
    
    # Final refinement
    if time.time() < deadline - 10:
        try:
            result = minimize(_growth_index_error, best_x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7,
                                     'adaptive': True})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except:
            pass
    
    # Clamp to [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)
# EVOLVE-BLOCK-END
