import numpy as np
from scipy import integrate, optimize
import time
from math import factorial
from scipy.special import eval_hermite

def _time_left(deadline):
    return deadline - time.time()

def _g_func(x, a0, a4, a8):
    H0_0 = eval_hermite(0, 0)  # 1
    H4_0 = eval_hermite(4, 0)  # 12
    H8_0 = eval_hermite(8, 0)  # 1344
    H12_0 = eval_hermite(12, 0)  # 665280
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    H0_val = eval_hermite(0, x)
    H4_val = eval_hermite(4, x)
    H8_val = eval_hermite(8, x)
    H12_val = eval_hermite(12, x)
    
    poly = a0 * H0_val + a4 * H4_val + a8 * H8_val + a12 * H12_val
    return poly * np.exp(-x**2 / 2.0)

def _objective_fast(params):
    a0, a4, a8 = params
    x = np.linspace(-10, 10, 10001)
    g_vals = _g_func(x, a0, a4, a8)
    return np.trapz(np.abs(g_vals), x)

def _objective_precise(params):
    a0, a4, a8 = params
    def integrand(x):
        return abs(_g_func(x, a0, a4, a8))
    
    x_grid = np.linspace(-10, 10, 5001)
    g_vals = _g_func(x_grid, a0, a4, a8)
    signs = np.sign(g_vals)
    zero_indices = np.where(np.diff(signs) != 0)[0]
    
    bounds = [-10.0]
    for idx in zero_indices:
        x1, x2 = x_grid[idx], x_grid[idx+1]
        g1, g2 = g_vals[idx], g_vals[idx+1]
        if abs(g2 - g1) > 1e-15:
            xz = x1 - g1 * (x2 - x1) / (g2 - g1)
            bounds.append(xz)
    bounds.append(10.0)
    
    total = 0.0
    for i in range(len(bounds)-1):
        val, err = integrate.quad(integrand, bounds[i], bounds[i+1], limit=200, epsabs=1e-12, epsrel=1e-12)
        total += val
    return total

def construct():
    deadline = time.time() + 170
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _objective_fast(seed)
    best_params = seed.copy()
    
    np.random.seed(123)
    
    starts = [seed.copy()]
    for scale in [0.1, 0.5, 1.0, 2.0]:
        starts.append(seed + scale * np.random.randn(3))
    for _ in range(15):
        starts.append(np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.01, 0.01)
        ]))
    
    for start in starts:
        if time.time() > deadline - 40:
            break
        try:
            res = optimize.minimize(
                _objective_fast, start, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    if time.time() < deadline - 30:
        try:
            res = optimize.minimize(
                _objective_precise, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    return np.array(best_params)
