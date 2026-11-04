import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 0):
            return 1e10 # Large penalty for invalid
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 0:
            return 1e10
            
        # Calculate cumulative angles (starting at 0)
        # phi[i] is the angle of point i
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate the minimum triangle area numerator
        # Area ~ sin(a) + sin(b) + sin(c) where a,b,c are arcs
        min_num = float('inf')
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    # Arcs between i-j, j-k, k-i
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        # Calculate hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 1e10
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    def run_optimization(x0):
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'xatol': 1e-8, 
                    'fatol': 1e-8, 
                    'maxiter': 1000, 
                    'adaptive': True
                }
            )
            return res.fun, res.x
        except Exception:
            return 1e10, x0

    # Initial guesses
    best_score = 1e10
    best_x = None
    
    # 1. Uniform start
    x0_uniform = np.full(N-1, 2 * np.pi / N)
    score, x = run_optimization(x0_uniform)
    if score < best_score:
        best_score = score
        best_x = x

    # 2. Random starts using Dirichlet distribution to ensure positive gaps summing to 1
    np.random.seed(42)
    for _ in range(10):
        raw = np.random.dirichlet(np.ones(N))
        x0 = raw[:N-1] * 2 * np.pi
        score, x = run_optimization(x0)
        if score < best_score:
            best_score = score
            best_x = x

    if best_x is None:
        best_x = x0_uniform

    opt_deltas = np.empty(N)
    opt_deltas[:N-1] = best_x
    opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
    
    # Ensure validity and fallback
    if np.any(opt_deltas <= 0) or best_score > 0: # best_score should be negative ratio
        opt_deltas = np.full(N, 2 * np.pi / N)

    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
