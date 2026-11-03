import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Generate initial Fibonacci spiral in Cartesian coordinates
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.column_stack((x, y, z))
    
    # Flatten to 1D array for optimizer
    x0 = pts.flatten()
    
    # Objective function
    def energy(params):
        pts = params.reshape(N, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Calculate pairwise distances
        # diff[i, j, k] = pts[i, k] - pts[j, k]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # We only need i < j
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        
        # Avoid log(0)
        if np.any(d < 1e-8):
            return 1e10
        
        # Energy = sum log(1/d) = -sum log(d)
        return -np.sum(np.log(d))
        
    # Optimize in Cartesian space
    # We don't need bounds because we normalize inside, but L-BFGS-B works best with bounds.
    # However, since we normalize, the magnitude doesn't matter for the energy, only the direction.
    # We can set loose bounds or none. L-BFGS-B requires bounds if not provided? No, it works without.
    # But to prevent overflow in exp/log during intermediate steps, bounds are good.
    # Let's use bounds [-10, 10] for each coordinate.
    bounds = [(-10, 10)] * (3 * N)
    
    res = minimize(energy, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if res.success or res.fun < energy(x0):
        best_params = res.x
    else:
        best_params = x0
        
    # Convert best_params back to Cartesian and normalize
    pts = best_params.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    return pts
