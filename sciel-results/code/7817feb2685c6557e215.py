import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.5, 0.5),
        (0.27, -1.0, 0.0),
    ]
    a_vals = np.linspace(0.05, 1.0, 40)
    
    def Om_of_a(a, Om0, w0, wa):
        a = np.atleast_1d(a)
        Ode0 = 1.0 - Om0
        E2 = Om0 * a**(-3) + Ode0 * np.exp(3.0 * wa * (1.0 - a))
        return Om0 * a**(-3) / E2
    
    def compute_reference_f(a, Om0, w0, wa):
        """Integrate growth ODE from a=1 backwards."""
        a = np.atleast_1d(a)
        n = len(a)
        x_grid = np.linspace(np.log(a[0]), 0.0, 3000)
        f_at_x = np.zeros_like(x_grid)
        f_at_x[-1] = 1.0
        
        for i in range(len(x_grid)-2, -1, -1):
            x = x_grid[i]
            h = x_grid[i+1] - x_grid[i]
            a_curr = np.exp(x)
            Om = Om_of_a(a_curr, Om0, w0, wa)[0]
            w_curr = w0 + wa * (1.0 - a_curr)
            Om_de = 1.0 - Om
            Hprime_H = 1.5 * (1.0 + w_curr) * Om_de
            
            def ode(xc, fc):
                ac = np.exp(xc)
                Om_c = Om_of_a(ac, Om0, w0, wa)[0]
                wc = w0 + wa * (1.0 - ac)
                Ode_c = 1.0 - Om_c
                HpH = 1.5 * (1.0 + wc) * Ode_c
                return -((1.0 + HpH) * fc - 1.5 * Om_c)
            
            k1 = ode(x, f_at_x[i+1])
            k2 = ode(x + h/2, f_at_x[i+1] + h/2 * k1)
            k3 = ode(x + h/2, f_at_x[i+1] + h/2 * k2)
            k4 = ode(x + h, f_at_x[i+1] + h * k3)
            f_at_x[i] = f_at_x[i+1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        f_vals = np.zeros(n)
        for j in range(n):
            xj = np.log(a[j])
            idx = np.searchsorted(x_grid, xj)
            if idx == 0:
                f_vals[j] = f_at_x[0]
            elif idx >= len(x_grid):
                f_vals[j] = f_at_x[-1]
            else:
                x0g, x1g = x_grid[idx-1], x_grid[idx]
                f0, f1 = f_at_x[idx-1], f_at_x[idx]
                t = (xj - x0g) / (x1g - x0g)
                f_vals[j] = f0 + t * (f1 - f0)
        return f_vals
    
    # Precompute reference values
    ref_data = []
    for (Om0, w0, wa) in cosmologies:
        f_ref = compute_reference_f(a_vals, Om0, w0, wa)
        Om_arr = Om_of_a(a_vals, Om0, w0, wa)
        ref_data.append((Om0, w0, wa, f_ref, Om_arr))
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def f_approx(coeffs, Om0, w0, wa):
        c0, c1, c2, c3, c4, c5 = coeffs
        Om = Om_of_a(a_vals, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        gamma = np.clip(gamma, 0.1, 1.2)
        return Om**gamma
    
    def objective(coeffs):
        max_err = 0.0
        for (Om0, w0, wa, f_ref, Om_arr) in ref_data:
            f_app = f_approx(coeffs, Om0, w0, wa)
            err = np.max(np.abs(f_app - f_ref) / np.abs(f_ref))
            max_err = max(max_err, err)
        return max_err
    
    x0 = np.array([0.55, 0.05, 0.0, 0.03, 0.0, 0.0])
    best_coeffs = x0.copy()
    best_score = objective(x0)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
        return best_coeffs.tolist()
    
    # Try multiple restarts if time allows
    restarts = [
        np.array([0.5, 0.1, -0.05, 0.05, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
    ]
    for xr in restarts:
        if time.time() - start_time > deadline - 5:
            break
        try:
            res = minimize(objective, xr, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    return best_coeffs.tolist()
