"""Compact dark-energy growth approximation with 6-coefficient gamma."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, quad

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _Om_matter(a, Om0, w0, wa):
    """Matter density parameter at scale factor a for CPL."""
    integral = (1.0+w0+wa)*np.log(a) - wa*(a-1.0)
    de_factor = np.exp(-3.0 * integral)
    return Om0 / (Om0 * a**(-3.0) + (1.0-Om0) * de_factor)

def _true_f(a, Om0, w0, wa):
    """True growth rate f(a) from GR ODE."""
    a_min, a_max = 0.01, 1.0
    x_min, x_max = np.log(a_min), 0.0
    
    def ode_rhs(x, y):
        a = np.exp(x)
        I = (1.0+w0+wa)*np.log(a) - wa*(a-1.0)
        E2 = Om0 * a**(-3.0) + (1.0-Om0) * np.exp(-3.0*I)
        I_prime = (1.0+w0+wa)/a - wa
        E2_prime = -3.0*Om0*a**(-4.0) - 3.0*(1.0-Om0)*np.exp(-3.0*I)*I_prime
        a_Hp_H = a * 0.5 * E2_prime / E2
        Om_a = Om0 / E2
        f = y[0]
        return [-f**2 - (2.0 + a_Hp_H)*f + 1.5*Om_a*a**2]
    
    sol = solve_ivp(ode_rhs, [x_min, x_max], [1.0], method='DOP853',
                    rtol=1e-9, atol=1e-11, dense_output=True, max_step=0.02)
    if not sol.success:
        return Om0**0.55
    x_eval = np.log(a)
    if x_eval < x_min or x_eval > x_max:
        return Om0**0.55
    return sol.sol(x_eval)[0]

def _true_D(a, Om0, w0, wa):
    """True normalized growth factor D(a)/D(1)."""
    integrand = lambda x: _true_f(np.exp(x), Om0, w0, wa)
    try:
        val, _ = quad(integrand, np.log(a), 0.0, limit=50)
        return np.exp(-val)
    except:
        return 1.0

def _approx_f(coeffs, a, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_matter(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    if Om_a <= 0:
        return 0.0
    return Om_a**gamma

def _approx_D(coeffs, a, Om0, w0, wa):
    integrand = lambda x: _approx_f(coeffs, np.exp(x), Om0, w0, wa)
    try:
        val, _ = quad(integrand, np.log(a), 0.0, limit=50)
        return np.exp(-val)
    except:
        return 1.0

def _max_error(coeffs, cosmologies, a_vals):
    max_err = 0.0
    for Om0, w0, wa in cosmologies:
        for a in a_vals:
            f_t = _true_f(a, Om0, w0, wa)
            D_t = _true_D(a, Om0, w0, wa)
            f_a = _approx_f(coeffs, a, Om0, w0, wa)
            D_a = _approx_D(coeffs, a, Om0, w0, wa)
            if abs(f_t) > 1e-10:
                max_err = max(max_err, abs(f_a - f_t)/abs(f_t))
            if abs(D_t) > 1e-10:
                max_err = max(max_err, abs(D_a - D_t)/abs(D_t))
    return max_err

def _find_coefficients():
    if _time_remaining() < 8.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    cosmologies = [(0.3, -1.0, 0.0), (0.3, -1.0, 0.4), (0.3, -1.0, -0.4),
                   (0.25, -1.0, 0.0), (0.35, -1.0, 0.0)]
    a_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    
    def obj(coeffs):
        coeffs = np.clip(coeffs, -10.0, 10.0)
        return _max_error(coeffs, cosmologies, a_vals)
    
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-5, 5)] * 6
    
    best = x0.copy()
    best_err = obj(x0)
    
    if _time_remaining() > 25.0:
        try:
            res = differential_evolution(obj, bounds, maxiter=40, popsize=10, seed=42)
            if res.fun < best_err:
                best, best_err = res.x, res.fun
        except:
            pass
    
    if _time_remaining() > 10.0:
        try:
            res2 = minimize(obj, best, method='Nelder-Mead', options={'maxiter': 150})
            if res2.fun < best_err:
                best, best_err = res2.x, res2.fun
        except:
            pass
    
    return [float(c) for c in np.clip(best, -10, 10)]

_COEFFS = _find_coefficients()

def construct():
    return _COEFFS
# EVOLVE-BLOCK-END
