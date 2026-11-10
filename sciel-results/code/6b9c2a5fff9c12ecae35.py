"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # 5s safety margin before 90s kill

# Cache for reference values
_REF_CACHE = {}


def _omega_m(a, Om0, w0, wa):
    """Compute matter density parameter Omega_m(a) for CPL dark energy."""
    if a <= 0:
        return 1.0
    Omd0 = 1.0 - Om0
    if abs(Omd0) < 1e-15:
        E2 = Om0 * a**(-3)
    else:
        # For CPL: w(a) = w0 + wa*(1-a)
        # integral from a to 1 of w(x) dx = w0*(1-a) + wa*(1-a)^2/2
        # DE factor: (1-Om0) * exp(3 * [w0*(1-a) + wa*(1-a)^2/2])
        # But in terms of a: E^2 = Om0*a^-3 + (1-Om0)*exp(3*integral_a^1 w dx)
        # integral_a^1 w(x)dx = w0*(1-a) + wa*(1-a)^2/2
        w_int = w0 * (1.0 - a) + wa * (1.0 - a)**2 / 2.0
        de_factor = (1.0 - Om0) * np.exp(3.0 * w_int)
        m_factor = Om0 * a**3
        E2 = (m_factor + de_factor) / a**3
    return Om0 * a**(-3) / E2


def _gamma_formula(a, Om0, w0, wa, coeffs):
    """Compute gamma from the six-coefficient formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma


def _f_formula(a, Om0, w0, wa, coeffs):
    """Compute f = Omega_m(a)^gamma from the formula."""
    gamma = _gamma_formula(a, Om0, w0, wa, coeffs)
    Om_a = _omega_m(a, Om0, w0, wa)
    if Om_a <= 0:
        return 0.0
    return Om_a ** gamma


def _reference_f_ode(a, Om0, w0, wa):
    """
    Compute reference f(a) = d ln D / d ln a using the second-order ODE.
    Uses the growth ODE: f' + f^2 + f*(3 + H'/H) - 1.5*Om_m = 0
    where H'/H = 1.5*(Om_m + (1+w)*Om_de)
    """
    key = (round(a, 10), round(Om0, 6), round(w0, 6), round(wa, 6))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    
    Om_a = _omega_m(a, Om0, w0, wa)
    
    # Initial condition at very early times
    a_init = 1e-4
    Om_init = _omega_m(a_init, Om0, w0, wa)
    # Use the matter-era solution: f ~ Om_m^(5/7) for early times
    f_init = Om_init ** (5.0/7.0)
    
    def ode_func(x, y):
        a_x = np.exp(x)
        if a_x <= 0 or a_x > 1.0:
            return [0.0]
        Om_x = _omega_m(a_x, Om0, w0, wa)
        Om_de_x = 1.0 - Om_x
        w_x = w0 + wa * (1.0 - a_x)
        Hprime_H = 1.5 * (Om_x + (1.0 + w_x) * Om_de_x)
        f_val = y[0]
        dfdx = -(f_val**2 + f_val * (3.0 + Hprime_H) - 1.5 * Om_x)
        return [dfdx]
    
    x_start = np.log(a_init)
    x_end = np.log(a)
    
    if x_end <= x_start:
        res = Om_a ** 0.55
        _REF_CACHE[key] = res
        return res
    
    try:
        sol = solve_ivp(
            ode_func, [x_start, x_end], [f_init],
            rtol=1e-8, atol=1e-10, max_step=0.05, method='RK45'
        )
        if sol.success and len(sol.y[0]) > 0:
            res = sol.y[0, -1]
        else:
            res = Om_a ** 0.55
    except Exception:
        res = Om_a ** 0.55
    
    _REF_CACHE[key] = res
    return res


def _reference_D_ratio(a, Om0, w0, wa):
    """
    Compute normalized growth factor D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    """
    key = ("D", round(a, 10), round(Om0, 6), round(w0, 6), round(wa, 6))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    
    if a >= 1.0:
        _REF_CACHE[key] = 1.0
        return 1.0
    
    # Use trapezoidal rule with fine grid for speed
    n_pts = 200
    x_vals = np.linspace(np.log(a), 0.0, n_pts)
    f_vals = np.array([_reference_f_ode(np.exp(x), Om0, w0, wa) for x in x_vals])
    integral = np.trapz(f_vals, x_vals)
    res = np.exp(-integral)
    
    _REF_CACHE[key] = res
    return res


def _precompute_references(cosmos, a_grid):
    """Pre-compute reference values for all models and grid points."""
    for Om0, w0, wa in cosmos:
        for a in a_grid:
            _reference_f_ode(a, Om0, w0, wa)
            _reference_D_ratio(a, Om0, w0, wa)


def _compute_max_relative_error(coeffs, cosmos, a_grid):
    """Compute the maximum relative error for f and D across all models and a values."""
    max_err = 0.0
    
    for Om0, w0, wa in cosmos:
        for a in a_grid:
            # f error
            f_form = _f_formula(a, Om0, w0, wa, coeffs)
            f_ref = _reference_f_ode(a, Om0, w0, wa)
            if abs(f_ref) > 1e-15:
                err_f = abs(f_form - f_ref) / abs(f_ref)
                max_err = max(max_err, err_f)
            
            # D error
            if a >= 1.0:
                D_form = 1.0
            else:
                n_pts = 200
                x_vals = np.linspace(np.log(a), 0.0, n_pts)
                f_form_vals = np.array([_f_formula(np.exp(x), Om0, w0, wa, coeffs) for x in x_vals])
                integral = np.trapz(f_form_vals, x_vals)
                D_form = np.exp(-integral)
            
            D_ref = _reference_D_ratio(a, Om0, w0, wa)
            if abs(D_ref) > 1e-15:
                err_D = abs(D_form - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
    
    return max_err


def _is_valid(coeffs, cosmos, a_grid):
    """Check coefficients are in [-10,10] and gamma in [0.1, 1.2]."""
    if len(coeffs) != 6:
        return False
    for c in coeffs:
        if c < -10 or c > 10 or not np.isfinite(c):
            return False
    
    for Om0, w0, wa in cosmos:
        for a in a_grid:
            gamma = _gamma_formula(a, Om0, w0, wa, coeffs)
            if gamma < 0.1 or gamma > 1.2 or not np.isfinite(gamma):
                return False
    
    return True


def _objective(coeffs, cosmos, a_grid):
    """Objective: max relative error, with penalty for invalid coefficients."""
    if not _is_valid(coeffs, cosmos, a_grid):
        return 100.0
    return _compute_max_relative_error(coeffs, cosmos, a_grid)


def construct():
    """Find optimal coefficients for the growth-index formula."""
    # Dense grid of cosmological parameters for fitting
    Om0_vals = np.array([0.25, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.2, -1.1, -1.0, -0.9, -0.8])
    wa_vals = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])
    
    # Create a stratified sample of cosmologies (not all combinations for speed)
    cosmos = []
    for i in range(len(Om0_vals)):
        for j in range(len(w0_vals)):
            for k in range(len(wa_vals)):
                cosmos.append((Om0_vals[i], w0_vals[j], wa_vals[k]))
    
    # Limit to a manageable subset for precomputation
    # Use a stratified sample
    n_cosmos = min(25, len(cosmos))
    indices = np.linspace(0, len(cosmos)-1, n_cosmos, dtype=int)
    cosmos_sample = [cosmos[i] for i in indices]
    
    # Scale factor grid
    a_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    # Pre-compute references
    _precompute_references(cosmos_sample, a_grid)
    
    # Initial guess: constant gamma = 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_obj = _objective(x0, cosmos_sample, a_grid)
    
    # Try multiple starting points
    starts = [
        x0,
        np.array([0.50, 0.10, -0.05, 0.05, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.52, 0.08, -0.03, 0.03, 0.02, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 10.0:
            break
        
        try:
            result = minimize(
                lambda c: _objective(c, cosmos_sample, a_grid),
                start,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-7, 'adaptive': True}
            )
            
            if result.fun < best_obj and _is_valid(result.x, cosmos_sample, a_grid):
                best_coeffs = result.x.copy()
                best_obj = result.fun
        except Exception:
            pass
    
    # Refine with least_squares if time permits
    if time.time() < _DEADLINE - 5.0:
        try:
            def residuals(coeffs):
                res_list = []
                for Om0, w0, wa in cosmos_sample:
                    for a in a_grid:
                        f_form = _f_formula(a, Om0, w0, wa, coeffs)
                        f_ref = _reference_f_ode(a, Om0, w0, wa)
                        if abs(f_ref) > 1e-15:
                            res_list.append((f_form - f_ref) / abs(f_ref))
                        
                        if a < 1.0:
                            n_pts = 100
                            x_vals = np.linspace(np.log(a), 0.0, n_pts)
                            f_form_vals = np.array([_f_formula(np.exp(x), Om0, w0, wa, coeffs) for x in x_vals])
                            integral = np.trapz(f_form_vals, x_vals)
                            D_form = np.exp(-integral)
                        else:
                            D_form = 1.0
                        D_ref = _reference_D_ratio(a, Om0, w0, wa)
                        if abs(D_ref) > 1e-15:
                            res_list.append((D_form - D_ref) / abs(D_ref))
                return np.array(res_list)
            
            result = least_squares(
                residuals,
                best_coeffs,
                bounds=([-10]*6, [10]*6),
                method='trf',
                max_nfev=200
            )
            
            if result.success and _is_valid(result.x, cosmos_sample, a_grid):
                obj_val = _compute_max_relative_error(result.x, cosmos_sample, a_grid)
                if obj_val < best_obj:
                    best_coeffs = result.x.copy()
                    best_obj = obj_val
        except Exception:
            pass
    
    # Final validation
    if not _is_valid(best_coeffs, cosmos_sample, a_grid):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
