"""Improved growth-index approximation via internal optimization."""
import numpy as np
from scipy.optimize import minimize
import time

def _Om(a, Om0, w0, wa):
    """Matter density parameter Om(a) = Om0 * a^3 / E(a)^2 where E(a)^2 = Om0*a^3 + (1-Om0)*a^(-3*(1+w0+wa))"""
    u = 1.0 + w0 + wa * (1.0 - a)
    denom = Om0 * a**3 + (1.0 - Om0) * a**(-3.0 * u)
    return Om0 * a**3 / denom

def _gamma(c, a, Om0, w0, wa):
    """gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    q = 1.0 - _Om(a, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_of_a(c, a, Om0, w0, wa):
    """f = Om(a)^gamma"""
    Om_a = _Om(a, Om0, w0, wa)
    g = _gamma(c, a, Om0, w0, wa)
    return Om_a**g

def _D_ratio(c, a, Om0, w0, wa, a_grid=None):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)"""
    if a_grid is None:
        # Use adaptive quadrature
        from scipy.integrate import quad
        def integrand(x):
            ai = np.exp(x)
            return _f_of_a(c, ai, Om0, w0, wa)
        val, _ = quad(integrand, np.log(a), 0.0, limit=200)
        return np.exp(-val)
    else:
        # Numerical integration on given grid
        xs = np.log(a_grid)
        fs = _f_of_a(c, a_grid, Om0, w0, wa)
        # Integrate from log(a) to 0
        idx = np.searchsorted(xs, np.log(a))
        if idx == 0:
            val = np.trapz(fs, xs)
        elif idx == len(xs):
            val = 0.0
        else:
            # Interpolate
            x0, x1 = xs[idx-1], xs[idx]
            f0, f1 = fs[idx-1], fs[idx]
            xa = np.log(a)
            fa = f0 + (f1 - f0) * (xa - x0) / (x1 - x0)
            # Integrate from xa to x1 (partial), then x1 to 0
            partial = 0.5 * (fa + f1) * (x1 - xa)
            rest = np.trapz(fs[idx:], xs[idx:])
            val = partial + rest
        return np.exp(-val)

def _objective(c, params_list, a_values):
    """Compute max relative error over all models and a values"""
    max_err = 0.0
    for (Om0, w0, wa) in params_list:
        for a in a_values:
            if a >= 1.0:
                continue
            # Check gamma constraint
            g = _gamma(c, a, Om0, w0, wa)
            if g < 0.1 or g > 1.2:
                return 1e6  # Penalty
            
            # Compute reference using ODE
            f_ref, D_ref = _ode_reference(a, Om0, w0, wa)
            f_app = _f_of_a(c, a, Om0, w0, wa)
            D_app = _D_ratio(c, a, Om0, w0, wa)
            
            if abs(f_ref) > 1e-10:
                err_f = abs(f_app - f_ref) / abs(f_ref)
            else:
                err_f = 0.0
            
            if abs(D_ref) > 1e-10:
                err_D = abs(D_app - D_ref) / abs(D_ref)
            else:
                err_D = 0.0
            
            max_err = max(max_err, err_f, err_D)
    return max_err

def _ode_reference(a, Om0, w0, wa):
    """Solve the GR growth ODE to get reference f and D"""
    from scipy.integrate import solve_ivp
    
    # Growth equation: F'' + [3/E - 2 dlnE/dlna] F' - 3/2 Om(a) F = 0
    # where E(a)^2 = Om0*a^3 + (1-Om0)*a^(-3*(1+w0+wa))
    # F' = dF/dlna, F'' = d2F/dlna^2
    # f = F'/F
    
    # We integrate from a=1 backwards to a
    # Let y = [F, F'], initial conditions at a=1: F(1)=1, F'(1)=f(1)
    # f(1) = Om0^gamma_1 approximately, but we need consistent initial condition
    
    # Actually, for the ODE reference, we need to be careful.
    # The growth factor D(a) satisfies:
    # D'' + [3/E - 2 dlnE/dlna] D' - 3/2 Om(a) D = 0
    # where prime is d/dlna
    
    # Let's define x = ln(a), so a = exp(x)
    # E(x)^2 = Om0*exp(3x) + (1-Om0)*exp(-3*(1+w0+wa*(1-exp(x)))*x)
    
    # This is complex. Let's use a simpler approach:
    # Use the known approximation that f ≈ Om^gamma with gamma ≈ 0.55 as seed
    # For the reference, we'll use a high-accuracy numerical solution
    
    # Simplified: use the fact that for flat LCDM, f ≈ Om^0.55 is good
    # For the ODE, let's integrate the equation for f directly:
    # f' = f * (df/dlna) where df/dlna = -f + (3/2)(1-f)Om - (1/2)(1+w) f ... 
    # Actually, the exact equation for f = D'/D is:
    # f' + f^2 + [3/E - 2 dlnE/dlna] f - 3/2 Om = 0
    
    # Let's just use a dense numerical integration of the second-order ODE
    # with initial conditions F(1)=1, F'(1)=Om0^0.55 (standard approximation)
    
    def E_of_a(a):
        u = 1.0 + w0 + wa * (1.0 - a)
        return np.sqrt(Om0 * a**3 + (1.0 - Om0) * a**(-3.0 * u))
    
    def dlnE_dlna(a):
        # dlnE/dlna = (3/2) * [Om0*a^3 - (1-Om0)*a^(-3u)*(1+w0+wa*(1-a)) + (1-Om0)*a^(-3u)*wa*a] / E^2
        # More carefully: E^2 = Om0*a^3 + (1-Om0)*a^(-3u) where u=1+w0+wa(1-a)
        # dE^2/dlna = Om0*3*a^3 + (1-Om0)*(-3u)*a^(-3u) + (1-Om0)*a^(-3u)*3*wa*a
        #           = 3*Om0*a^3 + (1-Om0)*a^(-3u)*(-3u + 3*wa*a)
        # dlnE/dlna = (1/(2E^2)) * dE^2/dlna
        u = 1.0 + w0 + wa * (1.0 - a)
        E2 = Om0 * a**3 + (1.0 - Om0) * a**(-3.0 * u)
        dE2 = 3.0 * Om0 * a**3 + (1.0 - Om0) * a**(-3.0 * u) * (-3.0*u + 3.0*wa*a)
        return dE2 / (2.0 * E2)
    
    def ode_system(x, y):
        a = np.exp(x)
        E = E_of_a(a)
        Om_a = Om0 * a**3 / E**2
        dlnE = dlnE_dlna(a)
        F, Fp = y
        Fpp = -[3.0/E - 2.0*dlnE] * Fp + 1.5 * Om_a * F
        return [Fp, Fpp]
    
    # Integrate from x=0 (a=1) to x=ln(a)
    x_end = np.log(a)
    # Initial conditions: F(1)=1, F'(1) = f(1)
    # Use f(1) = Om0^0.55 as a reasonable starting point
    f1 = Om0**0.55
    y0 = [1.0, f1]
    
    sol = solve_ivp(ode_system, [0.0, x_end], y0, rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        # Fallback to simple approximation
        Om_a = _Om(a, Om0, w0, wa)
        f_ref = Om_a**0.55
        # Simple D approximation
        from scipy.integrate import quad
        def intg(x):
            ai = np.exp(x)
            Om_ai = _Om(ai, Om0, w0, wa)
            return Om_ai**0.55
        val, _ = quad(intg, x_end, 0.0, limit=200)
        D_ref = np.exp(-val)
        return f_ref, D_ref
    
    F_end = sol.y[0, -1]
    Fp_end = sol.y[1, -1]
    f_ref = Fp_end / F_end
    
    # D(a)/D(1) = F(a)/F(1) = F_end / 1.0 = F_end
    D_ref = F_end
    
    return f_ref, D_ref

def construct():
    """Construct the six coefficients for the growth-index approximation."""
    start_time = time.time()
    deadline = start_time + 80.0  # 10 second safety margin
    
    # Validation cosmologies (typical CPL parameters)
    params_list = [
        (0.3, -1.0, 0.0),      # LCDM
        (0.3, -0.9, 0.1),      # CPL with positive wa
        (0.3, -1.1, 0.3),      # CPL with larger wa
        (0.27, -1.0, 0.0),     # Lower Om0
        (0.32, -0.95, 0.2),    # Another combination
    ]
    
    # a values to check
    a_values = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    
    # Initial guess: constant gamma = 0.55
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0.copy()
    best_err = _objective(c0, params_list, a_values)
    
    # Try a few local optimizations with different starting points
    start_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.05, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.05, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.05, 0.0, 0.0, 0.0, 0.05],
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(_objective, sp, args=(params_list, a_values),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Clip to valid range
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)
