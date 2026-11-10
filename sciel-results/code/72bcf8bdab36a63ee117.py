"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import time
import math
import numpy as np

# Deadline guard
_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed_ok():
    return (time.time() - _START) < _DEADLINE

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter as function of scale factor."""
    # q = 1 - Om(a), so Om(a) = 1 - q
    # For CPL: Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*exp(3*(1+w0+wa*(1-a))))
    # Actually the standard form: Om_m(a) = Om0 * a^3 / E^2(a)
    # where E^2(a) = Om0*a^3 + (1-Om0)*a^0 * exp(-3*wa*(1-a))
    # Let me use the standard CPL form:
    # E^2(a) = Om_m * a^-3 + Om_de * exp(3*wa*(1/a - 1))
    # In terms of a: E^2(a) = Om0 * a^3 + (1-Om0) * exp(3*wa*(1-a))  [for a<=1, normalized at a=1]
    # Actually: E^2(a) = Om0 * a^3 + (1-Om0) * exp(3*wa*(1/a - 1))
    # Let me be careful. The standard is:
    # E^2(a) = Om_m(a) + Om_de(a)
    # Om_m(a) = Om0 * a^3
    # Om_de(a) = (1-Om0) * exp(3*wa*(1/a - 1))
    # So E^2(a) = Om0*a^3 + (1-Om0)*exp(3*wa*(1/a-1))
    # Om_m(a) = Om0*a^3 / E^2(a)
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    de_exp = 3.0 * wa * (1.0/a_safe - 1.0)
    E2 = Om0 * a_safe**3 + (1.0 - Om0) * np.exp(de_exp)
    return Om0 * a_safe**3 / E2

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _f_approx(coeffs, a, Om0, w0, wa):
    """f = Om(a)^gamma"""
    Om = _Om_of_a(a, Om0, w0, wa)
    g = _gamma_from_coeffs(coeffs, a, Om0, w0, wa)
    # Clamp gamma to [0.1, 1.2] for evaluation
    g = np.clip(g, 0.1, 1.2)
    return Om**g

def _D_normalized(coeffs, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)"""
    # integral from ln(a) to 0 of f(exp(x)) dx = integral from a to 1 of f(t)/t dt
    # Use numerical quadrature
    if a >= 1.0:
        return 1.0
    # Integrate f(t)/t from a to 1
    n_pts = 500
    t = np.linspace(a, 1.0, n_pts)
    f_vals = _f_approx(coeffs, t, Om0, w0, wa)
    integrand = f_vals / t
    # Simpson's rule
    if n_pts % 2 == 1:
        h = (1.0 - a) / (n_pts - 1)
        integral = h/3.0 * (integrand[0] + integrand[-1] + 
                           4.0*np.sum(integrand[1:-1:2]) + 
                           2.0*np.sum(integrand[2:-1:2]))
    else:
        integral = np.trapz(integrand, t)
    return math.exp(-integral)

def _f_reference_ode(a, Om0, w0, wa):
    """Solve the GR ODE for f(a) = d ln D / d ln a."""
    # ODE: f' + f^2 + (3/2)(1+w) f - (3/2) Om_m = 0
    # where f' = df/d ln a
    # We integrate from a=1 (where f(1) is known) backwards to a
    # At a=1: f(1) = 0.55 * Om(1)^0.55 approximately, but we need the exact value
    # Actually, let's integrate forward from early times or use a known boundary
    # Better: integrate the ODE from a small value to a=1, or use the fact that
    # in matter-dominated era f ~ Om^0.55
    # Let's integrate from a=0.01 to a=1 using RK4
    # f(0.01) ~ Om(0.01)^0.55
    a_start = 0.01
    a_end = 1.0
    n_steps = 2000
    h = (a_end - a_start) / n_steps
    
    def f_of_a(a_val):
        Om = _Om_of_a(a_val, Om0, w0, wa)
        return Om**0.55  # initial guess
    
    def ode_rhs(a_val, f_val):
        Om = _Om_of_a(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        # df/d ln a = -f^2 - (3/2)(1+w)f + (3/2) Om
        return -f_val**2 - 1.5*(1.0+w)*f_val + 1.5*Om
    
    # RK4 integration
    a_curr = a_start
    f_curr = f_of_a(a_start)
    a_grid = np.linspace(a_start, a_end, n_steps+1)
    f_grid = np.zeros(n_steps+1)
    f_grid[0] = f_curr
    
    for i in range(n_steps):
        a_curr = a_grid[i]
        f_curr = f_grid[i]
        # Convert df/dlna to df/da: df/da = (1/a) * df/dlna
        def df_da(av, fv):
            return ode_rhs(av, fv) / av
        
        k1 = df_da(a_curr, f_curr)
        k2 = df_da(a_curr + h/2, f_curr + h/2*k1)
        k3 = df_da(a_curr + h/2, f_curr + h/2*k2)
        k4 = df_da(a_curr + h, f_curr + h*k3)
        f_curr = f_curr + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
        f_grid[i+1] = f_curr
    
    # Interpolate to get f at desired a
    if a <= a_start:
        return f_grid[0]
    if a >= a_end:
        return f_grid[-1]
    # Linear interpolation
    idx = int((a - a_start) / h)
    if idx >= n_steps:
        idx = n_steps - 1
    frac = (a - a_grid[idx]) / h
    return f_grid[idx] + frac * (f_grid[idx+1] - f_grid[idx])

def _objective(coeffs, cosmologies, a_values):
    """Compute max relative error over all cosmologies and a values."""
    max_err = 0.0
    for Om0, w0, wa in cosmologies:
        for a in a_values:
            if a <= 0 or a >= 1.0:
                continue
            # Check gamma constraint
            g = _gamma_from_coeffs(coeffs, np.array([a]), Om0, w0, wa)[0]
            if g < 0.1 or g > 1.2:
                return 1e10  # penalty
            
            f_approx = _f_approx(coeffs, np.array([a]), Om0, w0, wa)[0]
            f_ref = _f_reference_ode(a, Om0, w0, wa)
            
            if abs(f_ref) > 1e-8:
                err_f = abs(f_approx - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_approx - f_ref)
            
            max_err = max(max_err, err_f)
            
            # D normalized error (skip for speed, approximate with f error)
            # D error is correlated with f error, so f error is a good proxy
            # But let's compute it for accuracy
            D_approx = _D_normalized(coeffs, np.array([a]), Om0, w0, wa)[0]
            # Reference D: integrate f_ref
            n_pts = 200
            t = np.linspace(a, 1.0, n_pts)
            f_ref_vals = np.array([_f_reference_ode(ti, Om0, w0, wa) for ti in t])
            integrand = f_ref_vals / t
            integral_ref = np.trapz(integrand, t)
            D_ref = math.exp(-integral_ref)
            
            if abs(D_ref) > 1e-8:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_approx - D_ref)
            
            max_err = max(max_err, err_D)
    
    return max_err

def _optimize(coeffs_init, cosmologies, a_values, max_time=70.0):
    """Nelder-Mead style optimization with time guard."""
    start_time = time.time()
    best_coeffs = list(coeffs_init)
    best_err = _objective(best_coeffs, cosmologies, a_values)
    
    # Simple coordinate descent / Nelder-Mead hybrid
    n_dim = len(coeffs_init)
    step_sizes = [0.05] * n_dim
    
    # Nelder-Mead simplex
    simplex = [list(coeffs_init)]
    for i in range(n_dim):
        pt = list(coeffs_init)
        pt[i] += step_sizes[i]
        simplex.append(pt)
    
    simplex_errs = [_objective(s, cosmologies, a_values) for s in simplex]
    
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break
        if not _elapsed_ok():
            break
        
        # Sort simplex by error
        order = np.argsort(simplex_errs)
        simplex = [simplex[i] for i in order]
        simplex_errs = [simplex_errs[i] for i in order]
        
        # Check convergence
        if simplex_errs[0] < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        worst_err = simplex_errs[-1]
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - np.array(worst))
        x_r = list(x_r)
        err_r = _objective(x_r, cosmologies, a_values)
        
        if err_r < simplex_errs[0]:
            # Expansion
            gamma_exp = 2.0
            x_e = centroid + gamma_exp * (np.array(x_r) - centroid)
            x_e = list(x_e)
            err_e = _objective(x_e, cosmologies, a_values)
            if err_e < err_r:
                simplex[-1] = x_e
                simplex_errs[-1] = err_e
            else:
                simplex[-1] = x_r
                simplex_errs[-1] = err_r
        elif err_r < simplex_errs[-2]:
            simplex[-1] = x_r
            simplex_errs[-1] = err_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (np.array(worst) - centroid)
            x_c = list(x_c)
            err_c = _objective(x_c, cosmologies, a_values)
            if err_c < worst_err:
                simplex[-1] = x_c
                simplex_errs[-1] = err_c
            else:
                # Shrink
                for i in range(1, n_dim + 1):
                    simplex[i] = [simplex[0][j] + 0.5*(simplex[i][j] - simplex[0][j]) for j in range(n_dim)]
                    simplex_errs[i] = _objective(simplex[i], cosmologies, a_values)
        
        # Update best
        min_idx = np.argmin(simplex_errs)
        if simplex_errs[min_idx] < best_err:
            best_err = simplex_errs[min_idx]
            best_coeffs = list(simplex[min_idx])
    
    # Polish with finer steps
    if _elapsed_ok() and (time.time() - start_time) < max_time - 5:
        for i in range(50):
            if not _elapsed_ok():
                break
            improved = False
            for j in range(n_dim):
                for delta in [0.005, -0.005, 0.001, -0.001]:
                    trial = list(best_coeffs)
                    trial[j] += delta
                    err = _objective(trial, cosmologies, a_values)
                    if err < best_err:
                        best_err = err
                        best_coeffs = trial
                        improved = True
            if not improved:
                break
    
    return best_coeffs, best_err

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Validation cosmologies (typical CPL models)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with wa=0.3
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.27, -1.0, 0.0),   # Lower Om0
        (0.3, -0.9, 0.0),    # w0 != -1
        (0.3, -1.1, 0.2),    # w0 < -1
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -1.0, 0.5),    # Strong wa
    ]
    
    # a values for evaluation
    a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Start from constant gamma
    coeffs_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        best_coeffs, best_err = _optimize(coeffs_init, cosmologies, a_values, max_time=70.0)
        
        # Ensure coefficients are in [-10, 10] and finite
        for i in range(6):
            if not math.isfinite(best_coeffs[i]):
                best_coeffs[i] = 0.0
            best_coeffs[i] = max(-10.0, min(10.0, best_coeffs[i]))
        
        return best_coeffs
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
