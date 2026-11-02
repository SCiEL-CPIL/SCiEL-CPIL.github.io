import numpy as np
from scipy.optimize import differential_evolution

N = 50

# EVOLVE-BLOCK-START
def construct():
    best_h = None
    best_score = 0.0

    # We search for the best sparse support up to K=10
    # The objective is to maximize R = Q / (S * M)
    # Equivalent to minimizing J = S * M / Q
    # We fix S=1 by normalizing the weights in the objective function.
    
    def objective(weights):
        # weights are non-negative, sum to 1
        w = np.array(weights)
        if np.any(w < 0):
            return 1e10
        s = np.sum(w)
        if s == 0:
            return 1e10
        w = w / s
        
        q = np.sum(w**2)
        if q == 0:
            return 1e10
            
        conv = np.convolve(w, w, mode='full')
        m = np.max(conv)
        
        if m == 0:
            return 1e10
            
        # J = S * M / Q = 1 * M / Q
        return m / q

    # Try different K
    for K in range(2, 11):
        # Differential evolution bounds: [0, 1]
        bounds = [(0, 1)] * K
        
        # Run optimization
        try:
            res = differential_evolution(objective, bounds, maxiter=100, popsize=10, tol=1e-6, seed=42)
            if res.success:
                w_opt = res.x
                s = np.sum(w_opt)
                if s > 0:
                    w_opt = w_opt / s
                    q = np.sum(w_opt**2)
                    conv = np.convolve(w_opt, w_opt, mode='full')
                    m = np.max(conv)
                    score = q / m # Since S=1, R = Q/M
                    
                    if score > best_score:
                        best_score = score
                        best_h = w_opt
        except Exception:
            pass

    # If no optimization succeeded or score is low, fallback to known 2-element solution
    if best_h is None or best_score < 1.2:
        h = [0.0] * N
        h[0] = 2.0 / 3.0
        h[1] = 1.0 / 3.0
        return h

    # Pad best_h to length N
    h = [0.0] * N
    for i in range(len(best_h)):
        h[i] = best_h[i]
        
    return h
# EVOLVE-BLOCK-END
