"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    # Omega_m(a)
    Om = Om0 * a**(-3)
    # Omega_de(a)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    # H'/H = -3/2 [ Om + Ode*(1+3w) ]
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    # d2D/dx2 = -coeff * dD/dx + 1.5*Om*D
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _numerical_f_and_D(x_grid, Om0, w0, wa):
    # Solve ODE from x=ln(0.01) to x=0 (a=1), normalized D(1)=1
    x0 = np.log(0.01)
    # Solve forward with two basis solutions
    sol1 = solve_ivp(_ode_rhs, [x0, 0.0], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    sol2 = solve_ivp(_ode_rhs, [x0, 0.0], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    # Use sol2 as growing mode proxy
    D1 = sol2.sol(x_grid)[0]
    D1p = sol2.sol(x_grid)[1]
    Dn = D1 / D1[-1]
    fn = D1p / D1
    return fn, Dn

def _gamma_of(a, Om0, w0, wa, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = Om0 * a**(-3)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _compute_errors(coeffs, params, a_grid):
    """Compute max relative error in f and D over all params and a values."""
    x_grid = np.log(a_grid)
    max_err = 0.0
    for (Om0, w0, wa) in params:
        fn_num, Dn_num = _numerical_f_and_D(x_grid, Om0, w0, wa)
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        # Compute gamma from our formula
        gamma = _gamma_of(a_grid, Om0, w0, wa, coeffs)
        
        # Clip gamma to valid range
        gamma = np.clip(gamma, 0.1, 1.2)
        
        # Compute f from our formula
        valid_om = Om > 1e-10
        f_our = np.zeros_like(a_grid)
        f_our[valid_om] = Om[valid_om] ** gamma[valid_om]
        
        # Relative error in f
        valid_f = (fn_num > 0) & (f_our > 0) & valid_om
        if np.any(valid_f):
            err_f = np.abs(f_our[valid_f] - fn_num[valid_f]) / np.abs(fn_num[valid_f])
            max_err = max(max_err, np.max(err_f))
        
        # Compute D from our formula: D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # We need to integrate f(exp(x)) from x=ln(a) to x=0
        # Use numerical integration
        D_our = np.ones_like(a_grid)
        for i, ai in enumerate(a_grid):
            if ai >= 1.0:
                D_our[i] = 1.0
                continue
            xi = np.log(ai)
            # Integrate from xi to 0
            x_int = np.linspace(xi, 0.0, 100)
            a_int = np.exp(x_int)
            Om_int = Om0 * a_int**(-3)
            q_int = 1.0 - Om_int
            u_int = 1.0 + w0 + wa * (1.0 - a_int)
            gamma_int = _gamma_of(a_int, Om0, w0, wa, coeffs)
            gamma_int = np.clip(gamma_int, 0.1, 1.2)
            f_int = Om_int ** gamma_int
            # Integrate
            integral = np.trapz(f_int, x_int)
            D_our[i] = np.exp(-integral)
        
        # Relative error in D
        valid_D = (Dn_num > 0.01) & (D_our > 0.01)
        if np.any(valid_D):
            err_D = np.abs(D_our[valid_D] - Dn_num[valid_D]) / np.abs(Dn_num[valid_D])
            max_err = max(max_err, np.max(err_D))
    
    return max_err

def _fit_coefficients():
    # Representative CPL cosmologies - broader range
    params = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
        (0.27, -1.0, 0.0),
        (0.33, -1.0, 0.3),
    ]
    a_grid = np.logspace(-2, 0, 100)
    
    # Start with reasonable initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.05, 0.1, 0.0])
    
    # Use Nelder-Mead to minimize max error
    result = minimize(_compute_errors, x0, args=(params, a_grid),
                      method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.success:
        return result.x
    else:
        # Return initial guess if optimization fails
        return x0

def construct():
    # Optimize all 6 coefficients against ODE reference
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
