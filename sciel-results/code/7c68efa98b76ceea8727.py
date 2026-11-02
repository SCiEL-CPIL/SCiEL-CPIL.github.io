import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    best_h = None
    best_score = 0.0

    # Objective: Maximize R = Q / M
    # Equivalent to Minimizing J = M / Q
    # We normalize weights to sum to 1 inside the objective to handle scaling invariance.
    
    def objective(weights):
        w = np.array(weights)
        # Enforce non-negativity by clipping and re-normalizing
        # If all are zero, return large value
        if np.sum(w) == 0:
            return 1e10
        
        # Clip small negative values due to numerical noise
        w = np.clip(w, 0, None)
        
        s = np.sum(w)
        if s == 0:
            return 1e10
            
        w_norm = w / s
        
        q = np.sum(w_norm**2)
        if q < 1e-15:
            return 1e10
            
        conv = np.convolve(w_norm, w_norm, mode='full')
        m = np.max(conv)
        
        if m < 1e-15:
            return 1e10
            
        # Minimize M/Q
        return m / q

    def evaluate_solution(w):
        """Calculate the actual score R = Q/M for a normalized vector."""
        w = np.clip(np.array(w), 0, None)
        s = np.sum(w)
        if s == 0:
            return 0.0, None
        w_norm = w / s
        q = np.sum(w_norm**2)
        conv = np.convolve(w_norm, w_norm, mode='full')
        m = np.max(conv)
        if m == 0:
            return 0.0, None
        return q / m, w_norm

    # Strategy: Try multiple support sizes K.
    # For each K, run Nelder-Mead from several diverse initial points.
    
    max_K = 15
    initializations_per_K = 5
    
    for K in range(2, max_K + 1):
        # Generate diverse initial points
        # 1. Uniform
        # 2. Linear Ramp
        # 3. Concentrated at ends
        # 4. Random (fixed seed for reproducibility)
        # 5. Alternating
        
        inits = []
        
        # 1. Uniform
        inits.append(np.ones(K) / K)
        
        # 2. Linear Ramp (increasing)
        ramp = np.arange(1, K + 1)
        inits.append(ramp / np.sum(ramp))
        
        # 3. Concentrated at ends (High at 0 and K-1)
        ends = np.zeros(K)
        ends[0] = 0.5
        ends[-1] = 0.5
        inits.append(ends)
        
        # 4. Random
        rng = np.random.RandomState(42 + K)
        rand_init = rng.rand(K)
        inits.append(rand_init / np.sum(rand_init))
        
        # 5. Alternating (High, Low, High, Low...)
        alt = np.zeros(K)
        for i in range(K):
            if i % 2 == 0:
                alt[i] = 1.0
            else:
                alt[i] = 0.1
        inits.append(alt / np.sum(alt))
        
        for init in inits:
            try:
                # Nelder-Mead is robust for this small dimension
                res = minimize(objective, init, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.success or res.fun < 1e9:
                    w_opt = res.x
                    score, w_norm = evaluate_solution(w_opt)
                    
                    if score > best_score:
                        best_score = score
                        best_h = w_norm
            except Exception:
                pass

    # If no optimization succeeded or score is very low, fallback to a known good solution
    # The 2-element solution [2/3, 1/3] gives R = (4/9+1/9) / (4/9) = 5/4 = 1.25
    # Wait, let's calculate [2/3, 1/3]:
    # w = [2/3, 1/3]
    # Q = 4/9 + 1/9 = 5/9
    # Conv: [4/9, 4/9, 1/9] -> M = 4/9
    # R = (5/9) / (4/9) = 1.25
    
    if best_h is None or best_score < 1.25:
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
