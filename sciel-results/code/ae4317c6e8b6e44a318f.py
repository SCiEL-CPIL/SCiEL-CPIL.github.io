"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    m = 12  # number of double roots
    n = 1   # dimension
    
    # Initial guess: arithmetic progression like the seed
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Objective: minimize the C4 upper bound.
    # Based on Cohn-Goncalves construction, C4 is typically related to
    # the product of the double roots or a sum involving them.
    # Try: C4 = product of roots (to minimize, we want smaller roots)
    # But roots must be positive and satisfy constraints.
    # Alternative: C4 = sum(1/r_i) - minimizing this means larger roots
    # Let's try the product first, as smaller product likely gives better bound.
    
    def objective(x):
        # x is array of m positive root locations
        # C4 upper bound: try product of roots
        # To avoid overflow/underflow, use log-sum-exp
        if np.any(x <= 0):
            return 1e10
        return np.sum(np.log(x))  # log of product, minimizing this minimizes product
    
    # Constraints: roots must be positive and ordered
    # Use bounds for positivity
    bounds = [(1e-6, None)] * m
    
    # Constraints for ordering: x[i] < x[i+1]
    # SLSQP uses inequality constraints: fun(x) >= 0
    cons = []
    for i in range(m - 1):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i+1] - x[i] - 1e-6
        })
    
    # Also add constraint that roots are reasonably bounded
    # Upper bound to prevent unbounded solutions
    for i in range(m):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1000.0 - x[i]
        })
    
    # Run optimization
    best_x = x0.copy()
    best_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 60:  # safety margin
            if result.success and result.fun < best_val:
                best_x = result.x
                best_val = result.fun
    except Exception:
        pass
    
    # Ensure roots are positive and ordered
    best_x = np.maximum(best_x, 1e-6)
    for i in range(1, m):
        if best_x[i] <= best_x[i-1]:
            best_x[i] = best_x[i-1] + 1e-6
    
    # Convert to list of floats
    return [float(x) for x in best_x]
# EVOLVE-BLOCK-END
