import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 84x^10 + 2310x^8 - 27720x^6 + 138600x^4 - 277200x^2 + 11880
    
    def h0(x): return np.ones_like(x)
    def h4(x): return x**4 - 6*x**2 + 3
    def h8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def h12(x): return x**12 - 84*x**10 + 2310*x**8 - 27720*x**6 + 138600*x**4 - 277200*x**2 + 11880
    
    # Grid for integration
    # Use a sufficiently large range and fine grid for numerical integration
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    exp_term = np.exp(-x**2 / 2.0)
    
    # Precompute basis functions on the grid
    H0 = h0(x) * exp_term
    H4 = h4(x) * exp_term
    H8 = h8(x) * exp_term
    H12 = h12(x) * exp_term
    
    # Coefficients for H12 are determined by c0, c4, c8
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=11880
    # c12 = -(c0 + 3*c4 + 105*c8) / 11880
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 11880.0
        
        g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        
        # Check if g is identically zero (avoid division by zero)
        norm_sq = np.sum(g**2) * dx
        if norm_sq < 1e-12:
            return 1e10
            
        # Calculate <x^2> = integral(x^2 * g^2) / integral(g^2)
        x2_g2 = x**2 * g**2
        num = np.sum(x2_g2) * dx
        den = norm_sq
        
        return num / den

    # Initial guess: seed coefficients
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_result = None
    best_val = float('inf')
    
    # Try multiple random restarts if time permits
    # But first, try a single optimization from the seed
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline - 10:
        # Try a few random restarts
        rng = np.random.default_rng(42)
        for i in range(20):
            if time.time() > deadline - 10:
                break
            # Random perturbations around the seed or random small values
            # The coefficients are likely small.
            x_try = rng.normal(loc=x0, scale=0.1)
            try:
                res = minimize(objective, x_try, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x
            except Exception:
                continue
                
    if best_result is not None:
        return np.array(best_result)
    else:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
