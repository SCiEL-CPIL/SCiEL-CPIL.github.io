import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute constants
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 2032128.0

def _get_a12(a0, a4, a8):
    return -(a0 + 12.0*a4 + 1680.0*a8) / H12_0

def _compute_bound_squared(a0, a4, a8, x_grid):
    a12 = _get_a12(a0, a4, a8)
    # Calculate P(x)
    x2 = x_grid**2
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 73728x^10 + 580608x^8 - 2419200x^6 + 5443200x^4 - 5443200x^2 + 2032128
    
    # Use Horner's method or direct evaluation for speed
    # Let's just use numpy polynomial evaluation for clarity and sufficient speed
    H0 = np.ones_like(x_grid)
    H4 = 16*x2**2 - 48*x2 + 12
    H8 = 256*x2**4 - 3072*x2**3 + 11520*x2**2 - 13440*x2 + 1680
    H12 = 4096*x2**6 - 73728*x2**5 + 580608*x2**4 - 2419200*x2**3 + 5443200*x2**2 - 5443200*x2 + 2032128
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    f = P * np.exp(-0.5 * x2)
    max_val = np.max(np.abs(f))
    return max_val**2

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Use a sufficiently dense grid for the max calculation
    # The function is smooth and decays, so [-10, 10] with 2000 points is good.
    x_grid = np.linspace(-10, 10, 2001)
    
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.00030864])
    
    best_x = x0.copy()
    best_score = _compute_bound_squared(x0[0], x0[1], x0[2], x_grid)
    
    # Try several starting points to avoid local minima
    # The problem is 3-dimensional, so random restarts are cheap.
    rng = np.random.default_rng(42)
    
    starts = [x0]
    # Add some random perturbations
    for _ in range(10):
        perturbation = rng.normal(0, 0.1, size=3)
        starts.append(x0 + perturbation)
    
    # Also try a wide range
    for _ in range(5):
        starts.append(rng.uniform(-5, 5, size=3))
    
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        def objective(a):
            return _compute_bound_squared(a[0], a[1], a[2], x_grid)
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # One final refinement with L-BFGS-B if possible (though objective is non-smooth due to max)
    # Nelder-Mead is usually better for non-smooth objectives.
    
    return best_x
