import numpy as np
from scipy.optimize import differential_evolution

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    # The objective function takes the first N-1 gaps.
    # The N-th gap is determined by 2*pi - sum(x).
    # All gaps must be positive.
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 0):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 0:
            return 0.0
            
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
        # Hull area for points on unit circle is 0.5 * sum(sin(deltas))
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    # They must be positive and sum to 2*pi
    # A reasonable lower bound is small, upper bound is 2*pi - (N-1)*lower_bound
    # Let's use a small epsilon for lower bound
    lower = 1e-5
    upper = 2 * np.pi - (N-1) * lower
    
    # Use differential evolution for global optimization
    # This is more robust than Nelder-Mead for this problem
    result = differential_evolution(
        objective, 
        bounds=[(lower, upper)] * (N-1),
        seed=42,
        maxiter=1000,
        tol=1e-10,
        popsize=15,
        init='sobol'
    )
    
    if result.success:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
    else:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
