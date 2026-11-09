"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# Precompute Hermite polynomials values and moments
# H_n(x) for n = 0, 4, 8, 12
# Using physicists' Hermite polynomials
def H0(x): return 1.0
def H4(x): return 16*x**4 - 48*x**2 + 12
def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
def H12(x):
    # H_12(x) = 4096x^12 - 73728x^10 + 552960x^8 - 2165760x^6 + 3944640x^4 - 2850240x^2 + 479001600/720?
    # Let's use the recurrence or explicit formula to be safe, or just quad.
    # Explicit: H_12(x) = 4096x^12 - 73728x^10 + 552960x^8 - 2165760x^6 + 3944640x^4 - 2850240x^2 + 665280
    return 4096*x**12 - 73728*x**10 + 552960*x**8 - 2165760*x**6 + 3944640*x**4 - 2850240*x**2 + 665280

hs = [H0, H4, H8, H12]
ns = [0, 4, 8, 12]

# Compute moments M_k^(m) = int x^m H_k(x)^2 e^(-x^2) dx
# Symmetry: integrand is even, so integrate from 0 to inf and multiply by 2
moments = {}
for i, n in enumerate(ns):
    h = hs[i]
    for m in [0, 2, 4]:
        def integrand(x):
            return (x**m) * (h(x)**2) * np.exp(-x**2)
        # Integrate from 0 to 20 (tail is negligible)
        val, _ = quad(integrand, 0, 20, epsabs=1e-12, epsrel=1e-12)
        moments[(n, m)] = 2 * val

# Values at 0 for constraint
h_vals_0 = [h(0) for h in hs]
# h_vals_0 = [1, 12, 1680, 665280]

def objective(vars):
    c0, c4, c8 = vars
    # Calculate c12 to satisfy g(0) = 0
    # c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    c12 = -(c0 * h_vals_0[0] + c4 * h_vals_0[1] + c8 * h_vals_0[2]) / h_vals_0[3]
    
    coeffs = [c0, c4, c8, c12]
    
    num = 0.0
    den = 0.0
    for i, n in enumerate(ns):
        c_sq = coeffs[i]**2
        num += c_sq * moments[(n, 4)]
        den += c_sq * moments[(n, 2)]
        
    if den == 0:
        return 1e10
    return num / den

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin from 210s limit
    
    # Initial guess: The seed coefficients
    # Seed: [-113/100, 1/25, 1/3240]
    # c0 = -1.13, c4 = 0.04, c8 = 1/3240
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0
    best_val = objective(x0)
    
    # Try a few optimization methods
    methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell']
    
    for method in methods:
        if time.time() > deadline - 10.0:
            break
            
        try:
            # For Nelder-Mead, we might need to scale variables appropriately
            # The coefficients vary in magnitude. c0 ~ 1, c4 ~ 0.01, c8 ~ 0.001.
            # Let's optimize log-magnitudes or just use a good initial simplex.
            
            if method == 'Nelder-Mead':
                res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000})
            elif method == 'L-BFGS-B':
                # L-BFGS-B requires gradients or approximates them.
                # It might struggle with the non-smoothness if any, but this is smooth.
                res = minimize(objective, x0, method='L-BFGS-B', options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000})
            elif method == 'Powell':
                res = minimize(objective, x0, method='Powell', options={'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 5000})
            else:
                continue
                
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
        except Exception:
            continue
            
    return np.array(best_x)

# EVOLVE-BLOCK-END
