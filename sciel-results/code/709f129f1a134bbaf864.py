import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0 # Safety margin before 210s

# Precompute grid for fast numerical integration
# Gauss-Legendre quadrature on [0, R_MAX]
R_MAX = 15.0
N_QUAD = 200

nodes_gl, weights_gl = np.polynomial.legendre.leggauss(N_QUAD)
x_grid = 0.5 * R_MAX * (nodes_gl + 1.0)
w_grid = 0.5 * R_MAX * weights_gl

# Precompute weighted x^4 and exp(-x^2/2)
x4_w = (x_grid ** 4) * w_grid
exp_neg_half_x2 = np.exp(-0.5 * x_grid ** 2)

# Precompute Hermite polynomials at grid points
H0_grid = np.ones_like(x_grid)
H4_grid = eval_hermite(4, x_grid)
H8_grid = eval_hermite(8, x_grid)
H12_grid = eval_hermite(12, x_grid)

# H values at 0 for constraint
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 945.0

def get_c12(c0, c4, c8):
    return -(c0 + H4_0*c4 + H8_0*c8) / H12_0

def objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = get_c12(c0, c4, c8)
    
    P_grid = c0*H0_grid + c4*H4_grid + c8*H8_grid + c12*H12_grid
    g_grid = P_grid * exp_neg_half_x2
    
    # Smooth absolute value to enable gradient-based optimization
    eps = 1e-10
    abs_g_smooth = np.sqrt(g_grid**2 + eps)
    
    val = 2.0 * np.sum(x4_w * abs_g_smooth)
    return val

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try the seed first
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score

    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Fast objective enables many more starts
    num_starts = 300
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 20:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final polish with tighter tolerances
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(objective, best_coeffs, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs


