import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# Global variables for optimization
_cosmos = [
    (0.30, -1.0, 0.0),
    (0.30, -1.2, 0.5),
    (0.25, -0.9, 0.0),
    (0.35, -1.0, 0.3),
    (0.30, -1.05, 0.1),
    (0.28, -1.1, 0.2),
]
_a_grid = np.linspace(0.1, 1.0, 100)
_x_grid = np.log(_a_grid)

_ref_f = None
_ref_D = None

def _compute_references():
    global _ref_f, _ref_D
    ref_f_list = []
    ref_D_list = []
    
    for Om0, w0, wa in _cosmos:
        Om_de0 = 1.0 - Om0
        
        def dlnE_dx(x):
            term_m = -3.0 * Om0 * np.exp(-3.0 * x)
            term_de = -3.0 * (1.0 + w0 + wa) * Om_de0 * np.exp(-3.0 * (1.0 + w0 + wa) * x)
            e_sq = Om0 * np.exp(-3.0 * x) + Om_de0 * np.exp(-3.0 * (1.0 + w0 + wa) * x)
            return 0.5 * (term_m + term_de) / e_sq

        def Om_a(x):
            e_sq = Om0 * np.exp(-3.0 * x) + Om_de0 * np.exp(-3.0 * (1.0 + w0 + wa) * x)
            return Om0 * np.exp(-3.0 * x) / e_sq

        def ode_f(x, y):
            f_val = y[0]
            dlnE = dlnE_dx(x)
            Om = Om_a(x)
            return [ - (1.5 + dlnE) * f_val + 1.5 * Om ]

        x_start = -10.0
        x_end = 0.0
        sol = solve_ivp(ode_f, [x_start, x_end], [1.0], method='RK45', rtol=1e-8, atol=1e-10)
        
        f_interp = np.interp(_x_grid, sol.t, sol.y[0, :])
        
        J = np.zeros_like(_x_grid)
        for i in range(1, len(_x_grid)):
            dx = _x_grid[i] - _x_grid[i-1]
            J[i] = J[i-1] + 0.5 * (f_interp[i-1] + f_interp[i]) * dx
        D_vals = np.exp(J)
        
        ref_f_list.append(f_interp)
        ref_D_list.append(D_vals)
        
    _ref_f = np.array(ref_f_list)
    _ref_D = np.array(ref_D_list)

def _objective(coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    penalty = 0.0
    
    for i, (Om0, w0, wa) in enumerate(_cosmos):
        Om_de0 = 1.0 - Om0
        x = _x_grid
        e_m = Om0 * np.exp(-3.0 * x)
        e_de = Om_de0 * np.exp(-3.0 * (1.0 + w0 + wa) * x)
        e_sq = e_m + e_de
        Om = e_m / e_sq
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - np.exp(x))
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * np.exp(x) + c5 * u * q
        
        out_of_bounds = (gamma < 0.1) | (gamma > 1.2)
        if np.any(out_of_bounds):
            penalty += 100.0 * np.sum((np.clip(0.1 - gamma, 0, None) + np.clip(gamma - 1.2, 0, None))**2)
        
        f_approx = Om ** gamma
        
        J_approx = np.zeros_like(x)
        for j in range(1, len(x)):
            dx = x[j] - x[j-1]
            J_approx[j] = J_approx[j-1] + 0.5 * (f_approx[j-1] + f_approx[j]) * dx
        D_approx = np.exp(J_approx)
        
        err_f = np.abs(f_approx - _ref_f[i]) / np.abs(_ref_f[i])
        err_D = np.abs(D_approx - _ref_D[i]) / np.abs(_ref_D[i])
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
        
    return max_err + penalty

# Run optimization at import time
_start = time.time()
_compute_references()

x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
best_coeffs = x0

try:
    res = minimize(_objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
    if res.success and res.fun < _objective(x0):
        best_coeffs = res.x
except Exception:
    pass

# Clip to valid range
best_coeffs = np.clip(best_coeffs, -10, 10)

def construct():
    return list(best_coeffs)
