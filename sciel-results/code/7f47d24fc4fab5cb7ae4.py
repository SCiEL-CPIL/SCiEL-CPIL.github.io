import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 175.0
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 1032192.0
    
    def c12_from(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    x = np.linspace(-12, 12, 6000)
    xi = np.linspace(-12, 12, 6000)
    exp_neg_x2 = np.exp(-x**2 / 2)
    exp_neg_xi2 = np.exp(-xi**2 / 2)
    
    H0_x = np.ones_like(x)
    H4_x = eval_hermite(4, x)
    H8_x = eval_hermite(8, x)
    H12_x = eval_hermite(12, x)
    
    H0_xi = np.ones_like(xi)
    H4_xi = eval_hermite(4, xi)
    H8_xi = eval_hermite(8, xi)
    H12_xi = eval_hermite(12, xi)
    
    dx = x[1] - x[0]
    dxi = xi[1] - xi[0]
    
    def compute_C4(c0, c4, c8):
        c12 = c12_from(c0, c4, c8)
        
        g = (c0 * H0_x + c4 * H4_x + c8 * H8_x + c12 * H12_x) * exp_neg_x2
        A_f = np.sum(np.abs(g)) * dx
        
        g_hat = (c0 * H0_xi + c4 * H4_xi + c8 * H8_xi + c12 * H12_xi) * exp_neg_xi2
        A_fhat = np.sum(np.abs(g_hat)) * dxi
        
        return A_f * A_fhat
    
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    best_C4 = compute_C4(c0_init, c4_init, c8_init)
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    
    def objective(x):
        return compute_C4(x[0], x[1], x[2])
    
    x0 = np.array([c0_init, c4_init, c8_init])
    
    np.random.seed(42)
    starting_points = [x0]
    for _ in range(5):
        starting_points.append(x0 + 0.1 * np.random.randn(3))
    
    for sp in starting_points:
        if time.time() - start_time > deadline - 25:
            break
        try:
            result = minimize(objective, sp, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except:
            pass
    
    return best_coeffs
