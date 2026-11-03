"""Rewritten program: minimize max |autoconv| / (sum x)^2 for signed step functions."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 400

def construct():
    """
    Construct a sequence of N signed step heights minimizing
    max_k |autoconv_k| / (sum x)^2.
    
    The evaluator scores as seed/value where value is the raw objective (lower is better).
    We aim to minimize the objective.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds, keep safety margin
    
    # Objective: minimize max_k |c_k| / S^2 where c = x * x (autoconvolution), S = sum(x)
    # Let y = x / S, then sum(y) = 1 and objective = max_k |c_k| where c = y * y
    # So we want to minimize max_k |(y*y)_k| subject to sum(y) = 1.
    
    # This is a non-convex problem (rank-1 constraint implicit in y*y).
    # We'll use local optimization on y with sum(y)=1.
    
    # Initial guess: try various patterns
    best_val = np.inf
    best_y = None
    
    # Helper: compute autoconvolution and objective
    def obj(y):
        c = np.convolve(y, y, mode='full')
        return np.max(np.abs(c))
    
    # Try several initializations
    inits = []
    # 1. Uniform
    inits.append(np.ones(N) / N)
    # 2. Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    inits.append(alt / N)
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.standard_normal(N)
        inits.append(r / np.sum(r))
    # 4. Sparse-ish: few large values
    for _ in range(3):
        r = np.zeros(N)
        idx = rng.choice(N, size=10, replace=False)
        r[idx] = rng.choice([-1, 1], size=10)
        inits.append(r / np.sum(r))
    
    for y0 in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Use minimize with BFGS, handling the constraint sum(y)=1 via parameterization
        # Parameterize: y = y0 + t * v, where v is in the null space of sum(v)=0
        # Simpler: use scipy minimize with equality constraint
        
        def objective(y):
            c = np.convolve(y, y, mode='full')
            return np.max(np.abs(c))
        
        def gradient(y):
            c = np.convolve(y, y, mode='full')
            # Gradient of max|c_k| w.r.t. y
            # c_k = sum_i y_i y_{k-i}
            # dc_k/dy_j = y_{k-j} + y_{j-k} (with appropriate indexing)
            # This is complex; use numerical gradient or BFGS without gradient
            return None  # Let BFGS use numerical differentiation
        
        constraints = [{'type': 'eq', 'fun': lambda y: np.sum(y) - 1.0,
                       'jac': lambda y: np.ones(N)}]
        
        try:
            res = minimize(objective, y0, method='SLSQP',
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-8})
            if res.success or res.fun < best_val:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_y = res.x
        except Exception:
            pass
    
    # Convert back to x: x = y * S, but since objective is scale-invariant in the ratio,
    # and the evaluator measures max|autoconv| / (sum x)^2, we can return y directly
    # because for x = y, sum(x) = 1, so objective = max|autoconv(y)|.
    # Actually, let's return y as the step heights. The evaluator will compute
    # max|conv(y,y)| / (sum(y))^2 = max|conv(y,y)| / 1 = max|conv(y,y)|.
    
    if best_y is None:
        return np.ones(N)
    
    return best_y

# EVOLVE-BLOCK-END
