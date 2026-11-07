import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial guess
    h0 = np.full(N, 0.5)
    
    # Objective function
    def objective(h):
        # h is flattened
        h = np.clip(h, 0, 1) # Project to box? SLSQP handles bounds, but clipping inside objective can be non-smooth.
        # Better to rely on bounds.
        
        # Calculate correlations
        # correlate(h, 1-h, 'full')
        # We only care about the max.
        # To make it smooth, use log-sum-exp
        # c = np.correlate(h, 1-h, mode='full')
        # beta = 10.0
        # smooth_max = (1/beta) * np.log(np.sum(np.exp(beta * c)))
        # return (2.0 / N) * smooth_max
        
        # Let's try a simpler approach first: just the max, but SLSQP might struggle.
        # Let's use the smooth surrogate.
        c = np.correlate(h, 1-h, mode='full')
        # c can be negative?
        # h in [0,1], 1-h in [0,1]. Product is non-negative.
        # So c is non-negative.
        
        beta = 5.0
        # Avoid overflow
        c_max = np.max(c)
        smooth_max = c_max + (1/beta) * np.log(np.sum(np.exp(beta * (c - c_max))))
        
        return (2.0 / N) * smooth_max

    def objective_grad(h):
        # Numerical gradient is slow. Let's try analytical or just use finite diff if needed.
        # SLSQP can use finite diff, but it's slow.
        # Let's try to provide gradient.
        # f = (2/N) * (1/beta) * log(sum(exp(beta * c_k)))
        # c_k = sum_i h_i (1 - h_{i+k})
        # dc_k/dh_j = (1 - h_{j+k}) if j in range, and -h_{j-k} if j in range...
        # This is complex. Let's stick to finite difference or a very fast numerical gradient.
        # Or just use 'BFGS' which uses finite diff.
        # Given N=100, finite diff takes 100 evaluations per step.
        # SLSQP with finite diff might be too slow for many steps.
        
        # Let's try to optimize a simpler objective or use a different method.
        # Maybe just random search + local refinement?
        
        pass

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 50.0, 'jac': lambda h: np.ones(N)}
    ]
    
    bounds = [(0, 1) for _ in range(N)]
    
    # Try a few restarts with different initial guesses
    best_h = h0
    best_val = objective(h0)
    
    rng = np.random.default_rng(42)
    
    max_restarts = 10
    for i in range(max_restarts):
        if time.time() > deadline - 20: break
        
        # Generate initial guess
        # Add noise to 0.5
        h_init = h0 + rng.normal(0, 0.1, N)
        h_init = np.clip(h_init, 0, 1)
        # Adjust sum to be exactly 50
        diff = 50.0 - np.sum(h_init)
        h_init += diff / N
        h_init = np.clip(h_init, 0, 1)
        
        res = minimize(
            objective, 
            h_init, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-9}
        )
        
        if res.success:
            h_opt = res.x
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
                
    # Final check and return
    # Ensure constraints are met strictly for the returned value
    # The evaluator checks sum(h) == 50 within 1e-9 * 100.
    # SLSQP might leave a small error.
    # Let's do a final projection/correction.
    h_final = best_h.copy()
    
    # Project to sum 50
    # Simple correction
    diff = 50.0 - np.sum(h_final)
    h_final += diff / N
    h_final = np.clip(h_final, 0, 1)
    
    # If clipping changed the sum, we need to fix it again.
    # Iterative correction
    for _ in range(5):
        diff = 50.0 - np.sum(h_final)
        if abs(diff) < 1e-12:
            break
        h_final += diff / N
        h_final = np.clip(h_final, 0, 1)
        
    return h_final
