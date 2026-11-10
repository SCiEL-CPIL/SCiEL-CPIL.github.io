"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    Om = Om0 * a**(-3)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _get_growing_mode(x_grid, Om0, w0, wa):
    """Solve ODE and extract the physical growing mode properly."""
    x0 = np.log(0.01)
    # Solve with two basis solutions
    sol1 = solve_ivp(_ode_rhs, [x0, 0.0], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    sol2 = solve_ivp(_ode_rhs, [x0, 0.0], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol1.success or not sol2.success:
        return None, None
    
    D1 = sol1.sol(x_grid)[0]
    D1p = sol1.sol(x_grid)[1]
    D2 = sol2.sol(x_grid)[0]
    D2p = sol2.sol(x_grid)[1]
    
    # The growing mode should be dominant at late times (a->1, x->0)
    # At x=0, we want D to be positive and growing
    # Combine: D = A*D1 + B*D2. Choose A,B so that D is positive and smooth.
    # For CPL models, the growing mode is well-behaved. Use D2 as primary (initial slope 1)
    # and check if it's positive throughout. If not, adjust.
    
    D = D2
    Dp = D2p
    
    # If D becomes negative or zero, try combining with D1
    if np.any(D <= 0):
        # Try to find A such that A*D1 + D2 > 0 everywhere
        # At minimum, ensure positivity at the grid points
        min_ratio = np.min(D2[D1 > 1e-10] / D1[D1 > 1e-10]) if np.any(D1 > 1e-10) else -1
        A = max(0.0, -min_ratio + 0.1)  # Ensure positivity
        D = A * D1 + D2
        Dp = A * D1p + D2p
    
    # Normalize so D(1) = 1
    Dn = D / D[-1]
    fn = Dp / D
    
    # Ensure f is positive (it should be for growing mode)
    if np.any(fn <= 0):
        # Fallback: use D2 only if it works
        Dn = D2 / D2[-1]
        fn = D2p / D2
    
    return fn, Dn

def _compute_error(coeffs, params, a_grid):
    """Compute the max relative error of f and D for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    x_grid = np.log(a_grid)
    max_err = 0.0
    
    for (Om0, w0, wa) in params:
        fn_num, Dn_num = _get_growing_mode(x_grid, Om0, w0, wa)
        if fn_num is None:
            continue
        
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        
        # Check gamma constraint
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10  # Invalid
        
        # Compute f = Om^gamma
        valid = (Om > 1e-8) & (Om < 1.0)
        if np.sum(valid) < 10:
            continue
        
        fn_approx = Om[valid] ** gamma[valid]
        
        # Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
        # Use trapezoidal rule for the integral
        f_vals = fn_approx
        x_valid = x_grid[valid]
        
        # Integrate from ln(a) to 0
        D_approx = np.zeros_like(f_vals)
        for i in range(len(f_vals)):
            # Integral from x_valid[i] to 0 of f(exp(x)) dx
            # Use cumulative trapezoidal integration in reverse
            mask = x_valid >= x_valid[i]
            x_sub = x_valid[mask]
            f_sub = f_vals[mask]
            # Integrate from x_valid[i] to 0 (x_sub goes from x_valid[i] to ~0)
            # Reverse to integrate forward
            x_rev = x_sub[::-1]
            f_rev = f_sub[::-1]
            if len(x_rev) > 1:
                integral = np.trapz(f_rev, x_rev)
            else:
                integral = 0.0
            D_approx[i] = np.exp(-integral)
        
        # Relative errors
        err_f = np.abs(fn_approx - fn_num[valid]) / np.maximum(np.abs(fn_num[valid]), 1e-10)
        err_D = np.abs(D_approx - Dn_num[valid]) / np.maximum(np.abs(Dn_num[valid]), 1e-10)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return max_err

def _fit_coefficients():
    """Fit all 6 coefficients using Nelder-Mead optimization on the actual error metric."""
    from scipy.optimize import minimize
    
    # Representative CPL cosmologies for fitting
    params = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
    ]
    
    # Use a smaller grid for speed during optimization
    a_grid = np.logspace(-2, 0, 50)
    
    # Initial guess: constant gamma ~0.55 with small corrections
    x0 = np.array([0.55, 0.0, 0.0, 0.05, 0.1, 0.0])
    
    # Define objective function
    def objective(coeffs):
        # Enforce gamma constraint during optimization
        c0, c1, c2, c3, c4, c5 = coeffs
        # Check gamma bounds on a representative grid
        a_check = np.linspace(0.01, 1.0, 10)
        for (Om0, w0, wa) in params[:3]:  # Check on subset for speed
            Om = Om0 * a_check**(-3)
            q = 1.0 - Om
            u = 1.0 + w0 + wa * (1.0 - a_check)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_check + c5*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
        return _compute_error(coeffs, params, a_grid)
    
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if result.success or result.fun < 0.1:
        return result.x
    else:
        # Fallback to initial guess
        return x0

def construct():
    """Return six finite coefficients in [-10,10]."""
    try:
        coeffs = _fit_coefficients()
        # Clip to [-10,10] for safety
        coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
        # Sanity: ensure finite
        if not all(np.isfinite(c) for c in coeffs):
            raise ValueError
        return coeffs
    except Exception:
        # Fallback to a reasonable constant-gamma with mild CPL correction
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
# EVOLVE-BLOCK-END
