"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return best result before 630s limit
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    def laguerre_polynomial(x, m, n):
        """Compute Laguerre polynomial L_m^(n)(x)"""
        if m == 0:
            return np.ones_like(x)
        elif m == 1:
            return -(x + n)
        else:
            L_prev = np.ones_like(x)
            L_curr = -(x + n)
            for k in range(2, m + 1):
                L_new = ((2*k - 1 + n - x) * L_curr - (k - 1 + n) * L_prev) / k
                L_prev, L_curr = L_curr, L_new
            return L_curr
    
    def compute_bound(roots):
        """Compute the Cohn-Gonçalves upper bound for given root locations"""
        m = 12
        n = 1
        x_max = max(roots) + 10  # Extend beyond largest root
        
        # Create evaluation points
        x = np.linspace(0, x_max, 1000)
        
        # Compute the Laguerre polynomial combination
        # The refined construction involves a specific combination
        # For double roots, we need the polynomial to have sign changes
        # at each root location
        
        # Compute individual Laguerre polynomials
        L_m = laguerre_polynomial(x, m, n)
        
        # The upper bound is related to the integral of |L_m(x)|^2 * e^(-x)
        # But for sign uncertainty, we need to track sign changes
        
        # Count sign changes in the polynomial
        signs = np.sign(L_m)
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        
        # The bound involves the last sign change location
        # For m=12, we expect 12 sign changes (one at each double root)
        
        # Compute the actual bound value
        # This is proportional to the integral up to the last sign change
        weights = np.exp(-x)
        integral = np.trapz(L_m**2 * weights, x)
        
        # The upper bound C4 is related to this integral
        # Normalize by the total integral
        total_integral = np.trapz(weights, x)
        bound = integral / total_integral
        
        return bound
    
    # Initial guess: roots of generalized Laguerre polynomial L_12^(0)(x)
    # These are known to be optimal for the weighted L^2 norm
    try:
        roots_lag, _ = roots_genlaguerre(12, 0)
        x0 = np.sort(roots_lag)
        # Ensure positive and strictly increasing
        x0 = np.maximum(x0, 1e-6)
        if np.any(np.diff(x0) <= 1e-6):
            # Fallback to arithmetic progression if roots are too close
            x0 = np.array([4.0 * (i + 1) for i in range(12)])
    except Exception:
        x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Objective function to minimize
    def objective(roots):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # Timeout penalty
        
        # Ensure roots are positive and ordered
        if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
            return 1e10
        
        return compute_bound(roots)
    
    # Constraints: roots must be positive and strictly increasing
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 1e-6},
        {'type': 'ineq', 'fun': lambda x: x[1:] - x[:-1] - 1e-6}
    ]
    
    # Bounds: each root between 1e-6 and 100
    bounds = [(1e-6, 100)] * 12
    
    # Try SLSQP optimization with multiple starting points
    best_result = None
    best_score = np.inf
    
    # Multiple starting points: Laguerre roots, arithmetic progression, and perturbations
    start_points = [x0.copy()]
    
    # Add arithmetic progression
    start_points.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    # Add perturbed versions of Laguerre roots
    for scale in [0.5, 1.5, 2.0]:
        start_points.append(x0 * scale)
    
    for sp in start_points:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Ensure valid starting point
        if np.any(sp <= 0) or np.any(np.diff(sp) <= 1e-6):
            continue
        
        try:
            result = minimize(
                objective, 
                sp, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success and np.all(result.x > 0) and np.all(np.diff(result.x) > 0):
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_result = result.x.copy()
                    
        except Exception as e:
            continue
    
    if best_result is not None:
        return list(best_result)
    else:
        # Fall back to initial guess if optimization fails
        return list(x0)
    
# EVOLVE-BLOCK-END
