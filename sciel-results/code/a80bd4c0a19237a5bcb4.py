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
    # H_12(x) = 4096x^12 - 73728x^10 + 552960x^8 - 2165760x^6 + 3944640x^4 - 2850240x^2 + 665280
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
    
    # Reformulate as Chebyshev approximation / Remez-like approach
    # The objective is a Rayleigh quotient: (c^T N c) / (c^T D c)
    # where N and D are diagonal matrices of moments.
    # We want to minimize this ratio subject to the linear constraint for g(0)=0.
    
    # Let's try a more direct optimization using the analytic structure.
    # The variables are c0, c4, c8. c12 is fixed by the constraint.
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    # Let's use a global optimization method or a more robust local search.
    # The current Nelder-Mead might be getting stuck in local minima or moving slowly.
    # Let's try Differential Evolution or Basin Hopping if available, or just more robust Nelder-Mead with better scaling.
    
    # Scaling: c0 ~ 1, c4 ~ 0.04, c8 ~ 0.0003
    # Let's scale the variables to have similar magnitudes.
    # s0 = 1, s4 = 0.04, s8 = 0.001
    # y = x / s
    
    def objective_scaled(y):
        x = np.array([y[0]*1.0, y[1]*0.04, y[2]*0.001])
        return objective(x)
    
    y0 = np.array([x0[0]/1.0, x0[1]/0.04, x0[2]/0.001])
    
    best_y = y0
    best_val_scaled = objective_scaled(y0)
    
    # Try multiple methods on the scaled problem
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() > deadline - 20.0:
            break
            
        try:
            if method == 'Nelder-Mead':
                # Use a wider initial simplex to escape local minima
                # Nelder-Mead in scipy uses a default simplex.
                # We can provide our own if needed, but let's try defaults first with tight tolerances.
                res = minimize(objective_scaled, y0, method='Nelder-Mead', 
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(objective_scaled, y0, method='Powell', 
                               options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 10000})
            else:
                continue
                
            if res.fun < best_val_scaled:
                best_val_scaled = res.fun
                best_y = res.x
                
        except Exception:
            continue
            
    # Convert back to original variables
    best_x_final = np.array([best_y[0]*1.0, best_y[1]*0.04, best_y[2]*0.001])
    
    # Verify and update best
    final_val = objective(best_x_final)
    if final_val < best_val:
        best_val = final_val
        best_x = best_x_final
        
    # Additional local refinement from the best point found
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return np.array(best_x)

# EVOLVE-BLOCK-END
