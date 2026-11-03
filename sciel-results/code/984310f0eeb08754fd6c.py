import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Geometric parameterization: r_i = r0 * q^i for i=0..11
    # Optimize log(r0) and log(q) for better conditioning
    
    def roots_from_params(params):
        log_r0, log_q = params
        r0 = np.exp(log_r0)
        q = np.exp(log_q)
        # Ensure q > 1 for increasing roots, or handle both
        i = np.arange(12)
        roots = r0 * (q ** i)
        return roots
    
    def objective(params):
        """Minimize negative log of ||P||^2, where P is constructed from roots."""
        roots = roots_from_params(params)
        if np.any(roots <= 0) or not np.all(np.isfinite(roots)):
            return 1e10
        
        # Construct polynomial P(x) = prod (x - r_i)^2 for double roots
        # Evaluate in a Laguerre-weighted L2 norm
        # For Laguerre polynomials with alpha=0, weight is exp(-x)
        # Approximate ||P||^2 = integral_0^inf P(x)^2 * exp(-x) dx
        
        # Use numerical quadrature
        # P(x) = prod_{i=0}^{11} (x - r_i)^2
        # This is a degree-24 polynomial
        
        # Use Gauss-Laguerre quadrature with enough points
        # For degree 24, we need at least 13 points for exactness, use more for safety
        x, w = np.polynomial.laguerre.laggauss(80)
        
        # Compute P(x) at quadrature points
        # P(x) = prod (x - r_i)^2
        log_P = np.zeros_like(x)
        for r in roots:
            log_P += 2.0 * np.log(np.abs(x - r) + 1e-300)
        
        # Handle sign: P(x) = prod (x-r_i)^2 is always non-negative
        P_vals = np.exp(log_P)
        
        # Weighted L2 norm squared
        norm_sq = np.sum(w * P_vals)
        
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return 1e10
        
        # We want to maximize norm_sq, so minimize -log(norm_sq)
        return -np.log(norm_sq)
    
    # Multistart optimization
    best_val = 1e10
    best_params = None
    
    # Initial guesses in log-space
    # Seed roots are [4, 8, 12, ..., 48], arithmetic
    # Geometric approximation: r0 ~ 4, q ~ (48/4)^(1/11) ~ 12^(1/11) ~ 1.245
    np.random.seed(42)
    
    n_starts = 50
    for s in range(n_starts):
        if time.time() > deadline - 60:
            break
        
        if s == 0:
            # Start near the geometric approximation of the seed
            log_r0 = np.log(4.0) + np.random.randn() * 0.1
            log_q = np.log(1.245) + np.random.randn() * 0.05
        else:
            # Random starts
            log_r0 = np.random.uniform(np.log(0.5), np.log(20.0))
            log_q = np.random.uniform(np.log(1.05), np.log(2.5))
        
        x0 = np.array([log_r0, log_q])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=[(-5, 5), (-0.5, 1.5)],
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < best_val:
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Convert best params back to roots
    if best_params is not None:
        roots = roots_from_params(best_params)
        # Ensure all positive
        roots = np.maximum(roots, 1e-10)
        return [float(r) for r in roots]
    else:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
