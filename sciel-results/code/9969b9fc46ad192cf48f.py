import numpy as np
import time
from scipy.optimize import minimize

N = 600
# Domain [-0.25, 0.25]
x = np.linspace(-0.25, 0.25, N)
dx = x[1] - x[0]

def compute_score(f):
    # f is a numpy array of non-negative values
    if np.max(f) < 1e-9:
        return np.inf
    l1 = np.sum(f) * dx
    linf = np.max(f)
    l2_sq = np.sum(f**2) * dx
    
    # Autoconvolution peak is at 0 for non-negative f
    # (f*f)(0) = integral f(x)^2 dx = l2_sq
    # Normalization assumed to be l1 * linf
    # If this assumption is wrong, the score might be different, but we try to minimize the ratio
    # which is a common measure of "peakedness" or "concentration"
    
    if l1 * linf == 0:
        return np.inf
        
    return l2_sq / (l1 * linf)

def generate_f(params):
    a, b = params
    # f(x) = max(0, 1 - a*x^2 - b*x^4)
    f = 1.0 - a * x**2 - b * x**4
    f[f < 0] = 0.0
    return f

def objective(params):
    f = generate_f(params)
    return compute_score(f)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: Parabola
    # 1 - 2*x^2 is a triangle-ish shape?
    # For x in [-0.25, 0.25], x^2 max is 0.0625.
    # 1 - a*0.0625 = 0 => a = 16.
    # Let's start with a=10, b=0
    best_params = np.array([10.0, 0.0])
    best_score = objective(best_params)
    
    # Try a few initial points
    initial_guesses = [
        [10.0, 0.0],
        [16.0, 0.0], # Triangle
        [5.0, 50.0], # Sharper
        [0.0, 1000.0], # Very sharp
        [20.0, 100.0]
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Final check
    f = generate_f(best_params)
    
    # Ensure non-negative
    f = np.maximum(f, 0.0)
    
    return list(f)
