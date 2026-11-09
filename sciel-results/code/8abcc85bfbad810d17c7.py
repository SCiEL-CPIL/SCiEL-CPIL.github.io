"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
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
        """Compute the Cohn-Gonçalves upper bound for given root locations.
        
        The bound is the weighted position of the last sign change of a
        polynomial constructed to have double roots at the specified locations.
        Lower values indicate tighter uncertainty bounds.
        """
        m = 12
        n = 1
        
        # x_max must extend beyond the largest root to capture all sign changes
        x_max = max(roots) + 5.0
        
        # Create evaluation points (use enough points for accurate sign change detection)
        x = np.linspace(0, x_max, 2000)
        
        # Construct a polynomial with double roots at each specified location.
        # The polynomial is P(x) = prod_i (x - r_i)^2
        # This represents the Laguerre combination with prescribed double roots.
        P = np.ones_like(x)
        for r in roots:
            P *= (x - r)**2
        
        # Find sign changes in P(x)
        signs = np.sign(P)
        # Replace zeros to avoid issues
        signs[signs == 0] = 1.0
        
        # Identify sign change locations
        sign_change_mask = np.abs(np.diff(signs)) > 0
        if not np.any(sign_change_mask):
            # No sign changes found; return a large penalty
            return 1e6
        
        # Get indices of sign changes
        change_indices = np.where(sign_change_mask)[0]
        last_change_idx = change_indices[-1]
        
        # The last sign change location
        last_sign_change = x[last_change_idx]
        
        # The upper bound C4 is related to the weighted integral up to the last sign change
        # Use the exponential weight e^{-x} as in Laguerre orthogonality
        weights = np.exp(-x)
        
        # Compute the bound as the weighted mean position up to the last sign change
        # This gives a smooth, differentiable objective that depends on root locations
        x_weighted = x[:last_change_idx + 1] * weights[:last_change_idx + 1]
        w_total = np.sum(weights[:last_change_idx + 1])
        
        if w_total < 1e-15:
            return last_sign_change
        
        bound = np.sum(x_weighted) / w_total
        
        return bound
    
    # Initial guess: start with roots clustered near the origin to minimize the bound
    # The optimal configuration likely has roots spread out but starting early
    x0 = np.array([1.0 + 1.5 * i for i in range(12)])
    
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
    
    # Bounds: each root between 1e-4 and 50 (tighter upper bound since we want early roots)
    bounds = [(1e-4, 50)] * 12
    
    # Try SLSQP optimization with multiple restarts for better convergence
    best_result = None
    best_bound = float('inf')
    
    # Multiple initial guesses to escape local minima
    initial_guesses = [
        np.array([1.0 + 1.5 * i for i in range(12)]),
        np.array([0.5 + 1.0 * i for i in range(12)]),
        np.array([2.0 + 2.0 * i for i in range(12)]),
        np.array([1.0 + 0.8 * i for i in range(12)]),
    ]
    
    for x0_guess in initial_guesses:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # Stop if we're close to deadline
            break
        
        # Try SLSQP optimization
        try:
            result = minimize(
                objective, 
                x0_guess, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-10}
            )
            
            if result.success and np.all(result.x > 0) and np.all(np.diff(result.x) > 0):
                bound_val = compute_bound(result.x)
                if bound_val < best_bound:
                    best_bound = bound_val
                    best_result = result.x.copy()
                    
        except Exception:
            continue
    
    # If optimization found a valid result, return it
    if best_result is not None:
        return list(best_result)
    else:
        # Fall back to the best initial guess
        return list(initial_guesses[0])
    
# EVOLVE-BLOCK-END
