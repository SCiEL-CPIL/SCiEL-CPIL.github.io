"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _start_time

def _E2(a, Om0, Ode0, w0, wa):
    """H^2(a)/H0^2 for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, None)
    a3 = a**3
    de_exp = -3.0 * wa * (1.0 - a)
    de_term = Ode0 * a**(-3.0*(1.0+w0+wa)) * np.exp(de_exp)
    matter_term = Om0 / a3
    return matter_term + de_term

def _Om(a, Om0, Ode0, w0, wa):
    """Omega_matter(a)."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, None)
    a3 = a**3
    de_exp = -3.0 * wa * (1.0 - a)
    de_term = Ode0 * a**(-3.0*(1.0+w0+wa)) * np.exp(de_exp)
    matter_term = Om0 / a3
    return matter_term / (matter_term + de_term)

def _dE2_da(a, Om0, Ode0, w0, wa):
    """dE^2/da (not d/dln a)."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, None)
    a3 = a**3
    # matter: Om0 * a^-3, d/da = -3*Om0*a^-4
    dm = -3.0 * Om0 / a**4
    # DE: Ode0 * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
    alpha = -3.0*(1.0+w0+wa)
    de_base = Ode0 * a**alpha * np.exp(-3.0*wa*(1.0-a))
    dde = de_base * (alpha/a + 3.0*wa)
    return dm + dde

def _solve_growth_ode(a_grid, Om0, Ode0, w0, wa):
    """Solve growth ODE in x=ln a space. Returns f(x) and D(x)/D(1)."""
    x1 = np.log(a_grid[-1])  # a=1
    x0 = np.log(a_grid[0])
    
    def rhs(x, y):
        a = np.exp(x)
        E2v = _E2(a, Om0, Ode0, w0, wa)
        E = np.sqrt(E2v)
        dE2_da_val = _dE2_da(a, Om0, Ode0, w0, wa)
        dE_dx = 0.5 * dE2_da_val * a / E  # dE/dx = a * dE/da
        Omv = _Om(a, Om0, Ode0, w0, wa)
        # y = [D, D'] where D' = dD/dx
        D, Dp = y
        Dpp = -2.0 * dE_dx * Dp - 1.5 * Omv * E2v * D
        return [Dp, Dpp]
    
    # Integrate from x1 (a=1) backward to x0
    # Initial conditions at a=1: D(1)=1, D'(1) = f(1) ≈ Om0^gamma ~ 0.55
    # We'll use a reasonable initial slope
    y1 = [1.0, -0.55]  # going backward, D' is negative
    
    sol = solve_ivp(rhs, [x1, x0], y1, method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.05)
    
    if not sol.success:
        return None, None
    
    D_vals = sol.y[0, :]
    Dp_vals = sol.y[1, :]
    
    # Normalize: D(1) should be 1
    # The solution at x1 should give D=1
    D_norm = D_vals / D_vals[0]  # D_vals[0] is at x1=a=1
    f_vals = Dp_vals / D_vals  # f = dD/dx / D
    
    return f_vals, D_norm

def _compute_model_errors(coeffs, a_grid, cosmologies):
    """Compute max relative error for f and D across all cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for (Om0, Ode0, w0, wa) in cosmologies:
        f_ref, D_ref = _solve_growth_ode(a_grid, Om0, Ode0, w0, wa)
        if f_ref is None:
            continue
        
        # Compute model values
        Om = _Om(a_grid, Om0, Ode0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa*(1.0 - a_grid)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        
        # Check gamma range
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_model = Om**gamma
        
        # Compute D from model: D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # = exp(-integral from ln(a) to 0 of Om(e^x)^gamma dx)
        # Let's compute this numerically
        x_grid = np.log(a_grid)
        # Integrate f_model from x to 0
        # Use cumulative trapezoidal from the right
        f_x = f_model
        # integral from x_i to 0 = -integral from 0 to x_i
        # cumulative integral from left
        dx = np.diff(x_grid)
        cum_int = np.zeros_like(x_grid)
        cum_int[1:] = np.cumsum(0.5*(f_x[1:] + f_x[:-1]) * dx)
        # integral from x_i to 0 = -(cum_int[i] - cum_int[0]) ... 
        # Actually integral from 0 to x_i = cum_int[i] (since x_grid[0] is smallest)
        # integral from x_i to 0 = -cum_int[i]
        D_model = np.exp(-(-cum_int))  # = exp(cum_int)
        # Wait: D(a)/D(1) = exp(-∫_{ln a}^{0} f dx) = exp(∫_{0}^{ln a} f dx) = exp(cum_int)
        # But cum_int[0] = 0, and for a<1, ln a < 0, so cum_int[i] < 0 for i>0?
        # No: x_grid goes from small a to a=1, so x_grid[0] < x_grid[-1]=0
        # cum_int[i] = integral from x_grid[0] to x_grid[i]
        # We need integral from x_grid[i] to 0 = cum_int[-1] - cum_int[i]
        # cum_int[-1] = integral from x_grid[0] to 0
        integral_to_0 = cum_int[-1] - cum_int  # integral from x_i to 0
        D_model = np.exp(-integral_to_0)
        
        # Relative errors
        rel_f = np.abs(f_model - f_ref) / np.abs(f_ref)
        rel_D = np.abs(D_model - D_ref) / np.abs(D_ref)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return max_err

def construct():
    if _elapsed() > _DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Validation cosmologies (typical CPL models)
    cosmologies = [
        (0.3, 0.7, -1.0, 0.0),      # LCDM
        (0.3, 0.7, -1.0, 0.3),      # CPL with wa=0.3
        (0.3, 0.7, -0.9, -0.1),     # CPL variant
        (0.28, 0.72, -1.0, 0.1),    # another variant
        (0.31, 0.69, -1.05, 0.2),   # another
    ]
    
    a_grid = np.linspace(0.05, 1.0, 50)
    
    # Precompute references
    refs = []
    for cosmo in cosmologies:
        f_ref, D_ref = _solve_growth_ode(a_grid, *cosmo)
        refs.append((f_ref, D_ref))
    
    if _elapsed() > _DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def objective(coeffs):
        if _elapsed() > _DEADLINE:
            return 1e10
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for i, (Om0, Ode0, w0, wa) in enumerate(cosmologies):
            f_ref, D_ref = refs[i]
            if f_ref is None:
                continue
            
            Om = _Om(a_grid, Om0, Ode0, w0, wa)
            q = 1.0 - Om
            u = 1.0 + w0 + wa*(1.0 - a_grid)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
            
            f_model = Om**gamma
            
            x_grid = np.log(a_grid)
            dx = np.diff(x_grid)
            cum_int = np.zeros_like(x_grid)
            cum_int[1:] = np.cumsum(0.5*(f_model[1:] + f_model[:-1]) * dx)
            integral_to_0 = cum_int[-1] - cum_int
            D_model = np.exp(-integral_to_0)
            
            rel_f = np.abs(f_model - f_ref) / np.abs(f_ref)
            rel_D = np.abs(D_model - D_ref) / np.abs(D_ref)
            
            max_err = max(max_err, np.max(rel_f), np.max(rel_D))
        
        return max_err
    
    # Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best = x0.copy()
    best_obj = objective(x0)
    
    # Try multiple optimization strategies
    strategies = [
        {'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}},
        {'method': 'Powell', 'options': {'maxiter': 500, 'ftol': 1e-8}},
    ]
    
    for strat in strategies:
        if _elapsed() > _DEADLINE - 5:
            break
        try:
            res = minimize(objective, best, method=strat['method'], options=strat['options'])
            if res.fun < best_obj:
                best_obj = res.fun
                best = res.x
        except Exception:
            pass
    
    # Local refinement with small perturbations
    if _elapsed() < _DEADLINE - 10:
        for _ in range(3):
            if _elapsed() > _DEADLINE - 5:
                break
            for i in range(6):
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    trial = best.copy()
                    trial[i] += delta
                    if np.any(trial < -10) or np.any(trial > 10):
                        continue
                    obj = objective(trial)
                    if obj < best_obj:
                        best_obj = obj
                        best = trial
    
    # Clamp to [-10, 10]
    best = np.clip(best, -10.0, 10.0)
    
    return list(best)
# EVOLVE-BLOCK-END
