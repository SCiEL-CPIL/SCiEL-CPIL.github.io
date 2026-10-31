import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    N = 16
    D = 2
    n_dims = N * D
    
    def objective(x):
        pts = x.reshape(N, D)
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_min < 1e-8:
            return 1e10
            
        return (d_max / d_min)**2

    bounds = [(-1, 1) for _ in range(n_dims)]
    
    # Run DE
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=300, 
        popsize=20, 
        tol=1e-8, 
        seed=42,
        polish=True
    )
    
    best_x = result.x
    best_pts = best_x.reshape(N, D)
    
    # Optional: Local refinement
    # result2 = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 1000})
    # if result2.fun < result.fun:
    #     best_x = result2.x
    #     best_pts = best_x.reshape(N, D)
        
    return best_pts
