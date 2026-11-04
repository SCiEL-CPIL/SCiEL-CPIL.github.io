import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import time

def construct():
    start_time = time.time()
    deadline = 630.0 - 30.0  # 30s safety margin
    
    m = 12
    # Initial guess: seed configuration
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Objective: minimize C4. Assume C4 = 1 / prod(x_i)
    # So minimize -log(prod(x_i)) = -sum(log(x_i))
    def objective(x):
        if time.time() > deadline:
            return np.inf
        if np.any(x <= 0):
            return 1e10
        # Ensure ordering
        if np.any(np.diff(x) <= 0):
            return 1e10
        return -np.sum(np.log(x))
    
    # Constraints: x_i > 0, x_1 < x_2 < ... < x_12
    # Use bounds and ordering constraints
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple optimization runs
    for trial in range(50):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution
            x_start = best_x + np.random.randn(m) * 0.5
            x_start = np.sort(x_start)
            x_start = np.maximum(x_start, 0.1)
        
        # Bounds
        bounds = [(0.1, 100.0)] * m
        
        # Ordering constraints: x[i] < x[i+1] for all i
        def ordering_constraints(x):
            return np.diff(x) - 0.01  # Must be > 0
        
        constraints = [{'type': 'ineq', 'fun': ordering_constraints}]
        
        try:
            result = minimize(
                objective, x_start, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Return as list of floats
    return [float(x) for x in best_x]
