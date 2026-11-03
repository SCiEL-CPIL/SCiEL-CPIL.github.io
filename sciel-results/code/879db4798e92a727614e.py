import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s margin from 210s
    
    # Precompute Hermite polynomial values at 0 for the constraint
    # H_n(0) for even n: (-1)^(n/2) * n! / (n/2)!
    def h_even(n):
        if n % 2 != 0:
            return 0.0
        k = n // 2
        return ((-1) ** k) * (np.math.factorial(n) / np.math.factorial(k))
    
    h0_0 = h_even(0)
    h4_0 = h_even(4)
    h8_0 = h_even(8)
    h12_0 = h_even(12)
    
    # Grid for numerical integration
    # The function decays as e^{-pi x^2}, so x in [-10, 10] is plenty.
    # Step size 0.01 gives 2000 points.
    x = np.linspace(-10, 10, 2001)
    dx = x[1] - x[0]
    
    # Precompute weights
    w = dx * np.ones_like(x)
    
    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Evaluate P(x) = c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x)
        # Using physicists' Hermite polynomials
        P = (c0 * eval_hermite(0, x) + 
             c4 * eval_hermite(4, x) + 
             c8 * eval_hermite(8, x) + 
             c12 * eval_hermite(12, x))
        
        # f(x) = e^{-pi x^2} P(x)
        gaussian = np.exp(-np.pi * x**2)
        f = gaussian * P
        
        # Compute norms
        # ||f||_2^2
        norm_f_sq = np.sum(f**2) * dx
        if norm_f_sq < 1e-10:
            return 1e10
            
        # ||x f||_2^2
        norm_xf_sq = np.sum((x * f)**2) * dx
        
        # Compute derivative f'(x) numerically
        # Use central difference
        f_prime = np.zeros_like(f)
        f_prime[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
        f_prime[0] = (f[1] - f[0]) / dx
        f_prime[-1] = (f[-1] - f[-2]) / dx
        
        norm_fp_sq = np.sum(f_prime**2) * dx
        
        # Uncertainty product ratio
        # J = ||x f||_2 * ||f'||_2 / (2 pi ||f||_2^2)
        val = np.sqrt(norm_xf_sq * norm_fp_sq) / (2 * np.pi * norm_f_sq)
        
        return val

    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Run optimization
    best_val = objective(x0)
    best_x = x0
    
    try:
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B is good for smooth functions
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() - t_start > deadline:
        return np.array(best_x)
        
    # Try a few other random starts if time permits
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() - t_start > deadline - 5:
            break
        x_rand = rng.normal(0, 1, 3)
        # Scale to be in reasonable range
        x_rand = x_rand * 0.1
        try:
            res = minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return np.array(best_x)
