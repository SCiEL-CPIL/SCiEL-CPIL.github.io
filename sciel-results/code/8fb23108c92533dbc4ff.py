import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    # Initial guess: uniform distribution
    delta_init = np.full(N-1, 2 * np.pi / N)
    
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
        min_area = float('inf')
        
        # Vectorized calculation for speed
        for i in range(N):
            for j in range(i+1, N):
                ks = np.arange(j+1, N)
                if len(ks) == 0:
                    continue
                a = phi[j] - phi[i]
                b = phi[ks] - phi[j]
                c = 2 * np.pi - a - b
                # Area of triangle with vertices on unit circle
                areas = 0.5 * (np.sin(a) + np.sin(b) + np.sin(c))
                min_area = min(min_area, np.min(areas))
        
        # Calculate hull area denominator
        # Area of polygon inscribed in unit circle
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_area / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    # They must be positive and sum to 2*pi
    # A simple bound is (1e-5, 2*pi - 1e-5)
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_result = None
    best_score = -float('inf')
    
    # Use multiple restarts with L-BFGS-B for better convergence
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if restart == 0:
            x0 = delta_init.copy()
        else:
            # Add small noise to uniform distribution
            noise = rng.uniform(-0.1, 0.1, N-1)
            x0 = delta_init + noise
            # Ensure positivity and sum constraint roughly handled by optimizer, 
            # but initial guess should be valid
            x0 = np.clip(x0, 1e-5, 2 * np.pi - 1e-5)
            # Normalize x0 to sum to 2*pi - 1e-5 to be safe? 
            # Actually, L-BFGS-B handles bounds, but sum constraint is implicit.
            # If x0 sums to > 2*pi, last gap is negative -> invalid.
            # Let's scale x0 to sum to slightly less than 2*pi
            s = np.sum(x0)
            if s > 0:
                x0 = x0 * (2 * np.pi * 0.99) / s
                
        try:
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        except Exception:
            continue
            
    if best_result is not None:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(best_result.x)
    else:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
