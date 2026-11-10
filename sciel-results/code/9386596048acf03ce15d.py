"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad
import time

# EVOLVE-BLOCK-START
def _ode_rhs(t, y, Om0, w0, wa):
    """ODE for linear growth: y[0]=f, y[1]=ln(D/D(1))"""
    a = np.exp(t)
    Om_m = Om0 * a**(-3)
    Om_de = (1 - Om0) * a**(-3) * (1 + 3*w0 + 3*wa*(1-a))
    Om_k = 1 - Om0 - (1 - Om0)
    E2 = Om_m + Om_de + Om_k * a**(-2)
    Om_a = Om_m / E2
    
    f = y[0]
    lnD = y[1]
    
    # df/dln(a) = -f^2 - 3/2*(1 + Om_a)
    df = -f**2 - 1.5 * (1 + Om_a)
    # d(lnD)/dln(a) = -f  (since D(a)/D(1) = exp(-∫f dln(a)))
    dlnD = -f
    
    return [df, dlnD]

def _compute_true_f_D(Om0, w0, wa, a_values):
    """Compute true f and D(a)/D(1) via ODE."""
    t_min = np.log(a_values[0])
    t_max = 0.0
    
    # Initial conditions at a=1
    f1 = Om0**0.55
    y1 = [f1, 0.0]
    
    sol = solve_ivp(_ode_rhs, [t_max, t_min], y1, t_eval=np.log(a_values),
                    args=(Om0, w0, wa),
                    rtol=1e-10, atol=1e-12, method='DOP853')
    
    if not sol.success:
        return None, None
    
    f_true = sol.y[0]
    lnD_true = sol.y[1]
    D_true = np.exp(lnD_true)
    
    return f_true, D_true

def _compute_approx_f_D(coeffs, Om0, w0, wa, a_values):
    """Compute approximate f and D from coefficient formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    f_approx = np.zeros(len(a_values))
    lnD_approx = np.zeros(len(a_values))
    
    for i, a in enumerate(a_values):
        Om_m = Om0 * a**(-3)
        Om_de = (1 - Om0) * a**(-3) * (1 + 3*w0 + 3*wa*(1-a))
        Om_k = 1 - Om0 - (1 - Om0)
        E2 = Om_m + Om_de + Om_k * a**(-2)
        Om_a = Om_m / E2
        
        q = 1 - Om_a
        u = 1 + w0 + wa * (1 - a)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if gamma < 0.1 or gamma > 1.2:
            return None, None
        
        f_approx[i] = Om_a**gamma
    
    # Integrate to get D
    # D(a)/D(1) = exp(-∫_ln(a)^0 f(exp(x)) dx)
    # We need to integrate f_approx from 0 to ln(a) for each a
    t_vals = np.log(a_values)
    
    # Use cumulative trapezoidal integration
    # Integrate from t=0 backwards
    dt = np.diff(t_vals)
    # t_vals is decreasing, so we integrate from t=0 to t_vals[i]
    # ∫_0^{t_vals[i]} f(exp(x)) dx = -∫_{t_vals[i]}^0 f(exp(x)) dx
    
    # Let's compute integral from t_vals[i] to 0 for each i
    # Actually, let's just do numerical integration
    D_approx = np.zeros(len(a_values))
    D_approx[-1] = 1.0  # D(1) = 1
    
    for i in range(len(a_values) - 2, -1, -1):
        # Integrate from t_vals[i] to t_vals[i+1]
        # ∫_{t_vals[i]}^{t_vals[i+1]} f(exp(x)) dx
        # Using trapezoidal rule
        avg_f = 0.5 * (f_approx[i] + f_approx[i+1])
        integral = avg_f * (t_vals[i+1] - t_vals[i])
        # D(a_i)/D(1) = D(a_{i+1})/D(1) * exp(-integral)
        D_approx[i] = D_approx[i+1] * np.exp(-integral)
    
    return f_approx, D_approx

def _evaluate_max_error(coeffs, cosmologies, a_values):
    """Compute max relative error across all cosmologies and a values."""
    max_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        f_true, D_true = _compute_true_f_D(Om0, w0, wa, a_values)
        if f_true is None:
            continue
        
        f_approx, D_approx = _compute_approx_f_D(coeffs, Om0, w0, wa, a_values)
        if f_approx is None:
            return 1e6
        
        for i in range(len(a_values)):
            # Error in f
            if abs(f_true[i]) > 1e-10:
                err_f = abs(f_approx[i] - f_true[i]) / abs(f_true[i])
            else:
                err_f = abs(f_approx[i] - f_true[i])
            
            # Error in D
            if abs(D_true[i]) > 1e-10:
                err_D = abs(D_approx[i] - D_true[i]) / abs(D_true[i])
            else:
                err_D = abs(D_approx[i] - D_true[i])
            
            max_err = max(max_err, err_f, err_D)
    
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline with 15s margin
    
    # Representative CPL cosmologies covering parameter space
    cosmologies = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.30, -0.5, 0.0),
        (0.30, -1.5, 0.0),
        (0.25, -1.0, 0.3),
        (0.35, -1.0, -0.3),
        (0.30, -0.8, 0.2),
        (0.30, -1.2, -0.2),
    ]
    
    a_values = np.logspace(-2, 0, 40)
    
    # Precompute true values
    true_data = []
    for Om0, w0, wa in cosmologies:
        f_true, D_true = _compute_true_f_D(Om0, w0, wa, a_values)
        if f_true is not None:
            true_data.append((Om0, w0, wa, f_true, D_true))
    
    def objective(coeffs):
        if time.time() > deadline:
            return 1e6
        max_err = 0.0
        for Om0, w0, wa, f_true, D_true in true_data:
            f_approx, D_approx = _compute_approx_f_D(coeffs, Om0, w0, wa, a_values)
            if f_approx is None:
                return 1e6
            for i in range(len(a_values)):
                err_f = abs(f_approx[i] - f_true[i]) / max(abs(f_true[i]), 1e-10)
                err_D = abs(D_approx[i] - D_true[i]) / max(abs(D_true[i]), 1e-10)
                max_err = max(max_err, err_f, err_D)
        return max_err
    
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_coeffs = x0.copy()
    best_err = objective(x0)
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7,
                               'adaptive': True})
        if res.fun < best_err:
            best_coeffs = res.x
            best_err = res.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return [float(np.clip(c, -10, 10)) for c in best_coeffs]
    
    # Second refinement
    try:
        res2 = minimize(objective, best_coeffs, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9,
                                'adaptive': True})
        if res2.fun < best_err:
            best_coeffs = res2.x
            best_err = res2.fun
    except Exception:
        pass
    
    best_coeffs = np.clip(best_coeffs, -10, 10)
    return [float(c) for c in best_coeffs]
# EVOLVE-BLOCK-END
