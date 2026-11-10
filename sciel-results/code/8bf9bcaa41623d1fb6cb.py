"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, differential_evolution
import time

_START_TIME = time.time()
_DEADLINE = 82.0  # safety margin before 90s limit

# Cache for reference data
_ref_cache = {}

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_matter(a) for CPL dark energy."""
    if a == 1.0:
        return Om0
    
    def integrand_de(x):
        return (1.0 + w0 + wa * (1.0 - x)) / x
    
    if a < 1.0:
        integral_de, _ = quad(integrand_de, a, 1.0, limit=200)
    else:
        integral_de, _ = quad(integrand_de, 1.0, a, limit=200)
    
    ratio_de = np.exp(-3.0 * integral_de)
    ratio_m = a ** (-3.0)
    
    rho_m_a = Om0 * ratio_m
    rho_de_a = (1.0 - Om0) * ratio_de
    
    return rho_m_a / (rho_m_a + rho_de_a)

def _compute_reference_data(cosmologies, a_grid):
    """Compute dense reference data for f and D using ODE integration."""
    data = {}
    for Om0, w0, wa in cosmologies:
        key = (round(Om0, 10), round(w0, 10), round(wa, 10))
        if key in _ref_cache:
            data[key] = _ref_cache[key]
            continue
        
        # Compute Om(a) on the grid
        a_vals = np.array(a_grid)
        Om_vals = np.array([_Om_of_a(a, Om0, w0, wa) for a in a_vals])
        
        # Compute f from ODE
        lna_vals = np.log(a_vals)
        lna_start = -5.0
        y0 = [2.0/3.0]
        
        def ode_rhs(lna, y):
            a_local = np.exp(lna)
            Om_local = _Om_of_a(a_local, Om0, w0, wa)
            u_local = 1.0 + w0 + wa * (1.0 - a_local)
            dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
            Hprime_over_H = 0.5 * dlnH2_dlna
            f_local = y[0]
            df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
            return [df_dlna]
        
        # Integrate ODE on the grid
        sol = solve_ivp(ode_rhs, [lna_start, lna_vals[-1]], y0, 
                       t_eval=lna_vals, method='RK45', rtol=1e-9, atol=1e-11)
        
        if sol.success:
            f_vals = sol.y[0]
        else:
            # Fallback: compute f at each point individually
            f_vals = np.array([_f_reference_single(a, Om0, w0, wa) for a in a_vals])
        
        # Compute D(a)/D(1) from the reference f
        # D(a)/D(1) = exp(-integral from a to 1 of f(a') da'/a')
        # In terms of ln a: D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # We need to integrate f(exp(x)) dx from x to 0
        
        # Use cumulative integration
        lna_ref = lna_vals
        f_ref_vals = f_vals
        
        # Integrate from right to left (from a=1 to smaller a)
        # D(a)/D(1) = exp(-∫_a^1 f(a') da'/a') = exp(-∫_{ln a}^0 f(e^x) dx)
        # For a < 1: D(a)/D(1) = exp(∫_0^{ln a} f(e^x) dx) ... wait, let me be careful
        # Actually: d(ln D)/d(ln a) = f(a)
        # So ln D(a) - ln D(1) = ∫_1^a f(a') da'/a' = ∫_0^{ln a} f(e^x) dx
        # For a < 1, ln a < 0, so this is a negative integral
        # D(a)/D(1) = exp(∫_0^{ln a} f(e^x) dx)
        
        # Compute cumulative integral from 0 to each ln a
        # We have points at lna_vals, and we want ∫_0^{ln a_i} f(e^x) dx
        # For a_i < 1 (lna_i < 0): ∫_0^{lna_i} = -∫_{lna_i}^0
        
        # Let's compute ∫_{lna_i}^0 f(e^x) dx for each i
        # Using trapezoidal rule on the grid
        
        D_ratio = np.ones(len(a_vals))
        for i, lna_i in enumerate(lna_vals):
            if lna_i == 0.0:
                D_ratio[i] = 1.0
                continue
            if lna_i < 0.0:
                # ∫_{lna_i}^0 f(e^x) dx, then D = exp(-that)
                # Find points between lna_i and 0
                mask = (lna_ref >= lna_i) & (lna_ref <= 0.0)
                x_pts = lna_ref[mask]
                y_pts = f_ref_vals[mask]
                # Add endpoint if needed
                if len(x_pts) < 2:
                    D_ratio[i] = 1.0
                    continue
                integral = np.trapz(y_pts, x_pts)
                D_ratio[i] = np.exp(-integral)
            else:
                # ∫_0^{lna_i} f(e^x) dx, then D = exp(that)
                mask = (lna_ref >= 0.0) & (lna_ref <= lna_i)
                x_pts = lna_ref[mask]
                y_pts = f_ref_vals[mask]
                if len(x_pts) < 2:
                    D_ratio[i] = 1.0
                    continue
                integral = np.trapz(y_pts, x_pts)
                D_ratio[i] = np.exp(integral)
        
        _ref_cache[key] = (Om_vals, f_vals, D_ratio)
        data[key] = _ref_cache[key]
    
    return data

def _f_reference_single(a, Om0, w0, wa):
    """Reference growth rate from ODE for single point."""
    key = (round(a, 12), round(Om0, 12), round(w0, 12), round(wa, 12))
    if key in _ref_cache:
        return _ref_cache[key]
    
    Om_a = _Om_of_a(a, Om0, w0, wa)
    
    def ode_rhs(lna, y):
        a_local = np.exp(lna)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        u_local = 1.0 + w0 + wa * (1.0 - a_local)
        dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
        Hprime_over_H = 0.5 * dlnH2_dlna
        f_local = y[0]
        df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
        return [df_dlna]
    
    lna_target = np.log(a)
    lna_start = -5.0
    y0 = [2.0/3.0]
    
    sol = solve_ivp(ode_rhs, [lna_start, lna_target], y0, method='RK45', rtol=1e-8, atol=1e-10)
    if sol.success:
        result = sol.y[0, -1]
        _ref_cache[key] = result
        return result
    else:
        return 0.55

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _compute_max_error(coeffs, cosmologies, a_values, ref_data):
    """Compute max relative error for given coefficients against reference data."""
    max_rel_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        key = (round(Om0, 10), round(w0, 10), round(wa, 10))
        if key not in ref_data:
            continue
        Om_vals, f_vals, D_ratio = ref_data[key]
        
        for i, a in enumerate(a_values):
            Om_a = Om_vals[i]
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_ref = f_vals[i]
            
            if abs(f_ref) > 1e-10:
                rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_rel_err = max(max_rel_err, rel_err_f)
            
            D_ref = D_ratio[i]
            # Compute D_approx using the approximate gamma
            if a == 1.0:
                D_approx = 1.0
            else:
                # Integrate the approximate f
                def integrand_approx(x):
                    a_local = np.exp(x)
                    Om_local = _Om_of_a(a_local, Om0, w0, wa)
                    q_local = 1.0 - Om_local
                    u_local = 1.0 + w0 + wa * (1.0 - a_local)
                    gamma_local = _gamma_from_coeffs(coeffs, q_local, u_local, wa, a_local)
                    if gamma_local < 0.1 or gamma_local > 1.2:
                        return 0.55
                    return Om_local ** gamma_local
                
                lna_target = np.log(a)
                if lna_target < 0:
                    integral, _ = quad(integrand_approx, lna_target, 0.0, limit=100)
                    D_approx = np.exp(integral)
                else:
                    integral, _ = quad(integrand_approx, 0.0, lna_target, limit=100)
                    D_approx = np.exp(-integral)
            
            if abs(D_ref) > 1e-10:
                rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_rel_err = max(max_rel_err, rel_err_D)
    
    return 100.0 * max_rel_err

def _compute_sum_sq_error(coeffs, cosmologies, a_values, ref_data):
    """Compute sum of squared relative errors (for least-squares fitting)."""
    total_err = 0.0
    count = 0
    
    for Om0, w0, wa in cosmologies:
        key = (round(Om0, 10), round(w0, 10), round(wa, 10))
        if key not in ref_data:
            continue
        Om_vals, f_vals, D_ratio = ref_data[key]
        
        for i, a in enumerate(a_values):
            Om_a = Om_vals[i]
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_ref = f_vals[i]
            
            if abs(f_ref) > 1e-10:
                rel_err_f = (f_approx - f_ref) / f_ref
                total_err += rel_err_f ** 2
                count += 1
    
    return total_err / max(count, 1)

def construct():
    """Construct the six coefficients."""
    global _START_TIME, _DEADLINE
    
    # Define validation cosmologies and a-values
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
        (0.28, -1.0, 0.2),
        (0.32, -1.1, -0.3),
    ]
    
    # Dense a-grid for reference computation
    a_grid = np.linspace(0.05, 1.5, 50)
    
    # Compute reference data
    if time.time() > _DEADLINE - 30.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref_data = _compute_reference_data(cosmologies, a_grid)
    
    # For error evaluation, use a subset of a-values
    a_eval = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Initial guess: constant gamma = 0.55
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Compute initial error
    best_score = _compute_max_error(best_coeffs, cosmologies, a_eval, ref_data)
    
    # Phase 1: Differential evolution for global search
    if time.time() < _DEADLINE - 40.0:
        bounds = [(-2, 2), (-2, 2), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
        
        def de_objective(x):
            if time.time() > _DEADLINE - 5.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            # Use sum-squared error for smoother landscape
            return _compute_sum_sq_error(x, cosmologies, a_eval, ref_data)
        
        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=50, 
                                              popsize=15, tol=1e-6, seed=42,
                                              polish=False)
            if de_result.fun < best_score / 100.0:
                candidate = list(de_result.x)
                exact_score = _compute_max_error(candidate, cosmologies, a_eval, ref_data)
                if exact_score < best_score:
                    best_coeffs = candidate
                    best_score = exact_score
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead local optimization
    if time.time() < _DEADLINE - 20.0:
        def nm_objective(x):
            if time.time() > _DEADLINE - 5.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            return _compute_max_error(x, cosmologies, a_eval, ref_data)
        
        try:
            nm_result = minimize(nm_objective, best_coeffs, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True})
            if nm_result.fun < best_score:
                candidate = list(nm_result.x)
                if all(-10 <= c <= 10 for c in candidate):
                    exact_score = _compute_max_error(candidate, cosmologies, a_eval, ref_data)
                    if exact_score < best_score:
                        best_coeffs = candidate
                        best_score = exact_score
        except Exception:
            pass
    
    # Phase 3: Fine-tune with smaller steps
    if time.time() < _DEADLINE - 10.0:
        def nm_objective_fine(x):
            if time.time() > _DEADLINE - 3.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            return _compute_max_error(x, cosmologies, a_eval, ref_data)
        
        try:
            nm_result2 = minimize(nm_objective_fine, best_coeffs, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10,
                                         'adaptive': True, 'initial_simplex': None})
            if nm_result2.fun < best_score:
                candidate = list(nm_result2.x)
                if all(-10 <= c <= 10 for c in candidate):
                    exact_score = _compute_max_error(candidate, cosmologies, a_eval, ref_data)
                    if exact_score < best_score:
                        best_coeffs = candidate
                        best_score = exact_score
        except Exception:
            pass
    
    # Ensure coefficients are in valid range
    for i in range(6):
        if best_coeffs[i] < -10:
            best_coeffs[i] = -10
        elif best_coeffs[i] > 10:
            best_coeffs[i] = 10
    
    return best_coeffs
