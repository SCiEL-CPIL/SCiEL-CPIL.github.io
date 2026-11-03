import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Allow up to 110 seconds for optimization
    timeout = 110.0
    
    # Shifts to consider: 1 to N//2
    # We can subsample shifts to speed up constraints if needed, 
    # but N=100 is small enough to use all.
    shifts = np.arange(1, N // 2 + 1)
    num_shifts = len(shifts)
    
    # Precompute indices for speed
    # For each shift k, we need h[:N-k] and h[k:]
    # We'll compute A(k) on the fly in the constraint function
    
    def compute_A(h, k):
        # h is array
        return np.sum(h[:N-k] * h[k:]) * (2.0 / N)

    def objective(x):
        # x[0] is z, x[1:] is h
        z = x[0]
        # We want to maximize z, so minimize -z
        return -z

    def constraint_sum(x):
        h = x[1:]
        return np.sum(h) - N / 2.0

    # We will create a single constraint function that returns a vector
    # SLSQP supports vector constraints if 'fun' returns an array
    def constraints_vec(x):
        z = x[0]
        h = x[1:]
        cons = np.zeros(num_shifts)
        for i, k in enumerate(shifts):
            cons[i] = compute_A(h, k) - z
        return cons

    # Initial guess
    h0 = np.full(N, 0.5)
    # Estimate initial z
    # For constant h, A(k) = 0.5 * (1 - k/N) * 2? 
    # A(k) = sum(0.25) * 2/N = (N-k)*0.25 * 2/N = 0.5 * (1 - k/N)
    # Min A(k) is at k=N/2, A = 0.5 * (1 - 0.5) = 0.25
    z0 = 0.2
    x0 = np.concatenate(([z0], h0))
    
    bounds = [(0, 1)] * (N + 1)
    
    # Constraints
    # Equality: sum(h) = N/2
    # Inequality: A(k) - z >= 0 for all k
    cons = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraints_vec}
    ]
    
    try:
        # Run SLSQP
        # maxiter might need to be higher, but time limit protects us
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
        
        if res.success:
            h_opt = res.x[1:]
            # Clean up h
            h_opt = np.clip(h_opt, 0, 1)
            # Ensure sum is exactly N/2
            current_sum = np.sum(h_opt)
            diff = N / 2.0 - current_sum
            if abs(diff) > 1e-9:
                # Add diff/N to each, then clip again
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
                # If clipping changed sum, do a final adjustment on the largest elements
                current_sum = np.sum(h_opt)
                diff = N / 2.0 - current_sum
                if abs(diff) > 1e-9:
                    # Distribute diff to the elements with most room
                    # Simple approach: add to all, clip, repeat
                    for _ in range(10):
                        h_opt += diff / N
                        h_opt = np.clip(h_opt, 0, 1)
                        current_sum = np.sum(h_opt)
                        diff = N / 2.0 - current_sum
                        if abs(diff) < 1e-9:
                            break
            return h_opt
        else:
            # If optimization failed, return seed
            return np.full(N, 0.5)
            
    except Exception:
        return np.full(N, 0.5)

# EVOLVE-BLOCK-END
