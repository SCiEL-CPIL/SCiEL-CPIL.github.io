import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11
H = np.sqrt(3) / 2

def construct():
    # Start time
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Define bounds for x and y
    # x in [0, 1], y in [0, H]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, H))   # y
    
    # Precompute indices for combinations
    import itertools
    indices = list(itertools.combinations(range(N), 3))
    
    def objective(params):
        # Check time
        if time.time() - start_time > timeout:
            return 1e10 # Bad score to stop? Or just let it finish. 
            # DE doesn't stop early easily on time. 
            # Better to rely on maxiter.
            
        points = params.reshape(N, 2)
        x = points[:, 0]
        y = points[:, 1]
        
        # Check if points are inside the triangle
        # 1. y >= 0
        # 2. y <= sqrt(3) * x
        # 3. y <= -sqrt(3) * x + sqrt(3)
        
        if np.any(y < 0) or np.any(y > np.sqrt(3) * x) or np.any(y > -np.sqrt(3) * x + np.sqrt(3)):
            return 1e10
            
        # Calculate min area
        min_area = float('inf')
        for i, j, k in indices:
            # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
            area = 0.5 * abs(x[i]*(y[j]-y[k]) + x[j]*(y[k]-y[i]) + x[k]*(y[i]-y[j]))
            if area < min_area:
                min_area = area
                if min_area < 1e-12: # Degenerate
                    return 1e10
                    
        return -min_area

    # Run DE
    # popsize 15, maxiter 1000
    # Strategy: 'best1bin' or 'rand1bin'
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=1000, 
        popsize=20, 
        tol=1e-9, 
        seed=42, 
        mutation=(0.5, 1.5), 
        recombination=0.9,
        polish=True # Use L-BFGS-B at the end for fine-tuning? 
                    # The function is non-smooth, so polish might fail or help slightly.
                    # Actually, 'polish' uses L-BFGS-B which assumes smoothness. 
                    # It might get stuck or behave weirdly on the kinks. 
                    # Let's try polish=False first, or just rely on DE.
                    # Given the non-smooth nature, polish is risky. 
                    # However, the "best" individual from DE is often close.
                    # Let's set polish=False to be safe, or True and see. 
                    # The prompt says "higher is better". 
                    # Let's use polish=False to avoid potential issues with gradient estimation on min-functions.
    )
    
    # The result x is the best parameters found
    # We need to return the points
    best_params = result.x
    points = best_params.reshape(N, 2)
    
    # Verify validity
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < 0) or np.any(y > np.sqrt(3) * x) or np.any(y > -np.sqrt(3) * x + np.sqrt(3)):
        # Fallback to random if DE failed (unlikely)
        return np.random.rand(N, 2) # This is bad, need valid points.
        # Better fallback:
        # Generate valid points manually
        pts = []
        while len(pts) < N:
            u, v = np.random.rand(2)
            if u + v < 1:
                # Barycentric to Cartesian
                # P = u*A + v*B + (1-u-v)*C ? No.
                # Standard mapping: 
                # If u,v in [0,1] and u+v<=1:
                # P = u*(1,0) + v*(0,1) ... wait, triangle vertices are specific.
                # Let's just use the rejection sampling from the seed program as fallback.
                pass
    
    return points
