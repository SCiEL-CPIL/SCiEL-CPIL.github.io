import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    
    # Initial guess: Hexagonal lattice
    # We want 16 points. A 4x4 grid is simple.
    # Let's try to arrange them in a roughly circular cluster.
    # Hexagonal coordinates:
    # i from 0 to 3, j from 0 to 3
    # x = i + 0.5 * (j % 2)
    # y = j * np.sqrt(3) / 2
    
    points = np.zeros((n, 2))
    idx = 0
    for j in range(4):
        for i in range(4):
            points[idx, 0] = i + 0.5 * (j % 2)
            points[idx, 1] = j * np.sqrt(3) / 2.0
            idx += 1
            
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Define objective function to minimize
    # We want to maximize R = (d_min / d_max)^2
    # So we minimize -R or 1/R. Let's minimize 1/R to keep it positive and bounded away from 0 if possible?
    # Actually, minimizing 1/R is equivalent to maximizing R.
    # If d_max = 0, R=0. 1/R is inf.
    # Let's define the cost as -np.log(R) if R>0 else large number.
    # Or simply 1.0 / R.
    
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 1e10 # Penalize zero distance
        if d_min == 0:
            return 1e10 # Penalize coincident points
            
        ratio = d_min / d_max
        # We want to maximize ratio.
        # Minimizing 1/ratio is a good convex-like surrogate (though not strictly convex)
        return 1.0 / (ratio ** 2) # This is 1/R
        
    # Flatten the initial points
    x0 = points.flatten()
    
    # Use Nelder-Mead for local optimization
    # Nelder-Mead doesn't require gradients and works well for low-dimensional non-smooth-ish functions (min/max are non-smooth, but the landscape is manageable)
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    best_points = result.x.reshape(n, 2)
    
    # Optional: Try a few random restarts or different initializations if time permits?
    # 600s is a lot. Nelder-Mead on 32 variables should be fast.
    # Let's run a second optimization starting from a square grid just in case.
    
    points_sq = np.zeros((n, 2))
    idx = 0
    for i in range(4):
        for j in range(4):
            points_sq[idx, 0] = i
            points_sq[idx, 1] = j
            idx += 1
    points_sq -= np.mean(points_sq, axis=0)
    x0_sq = points_sq.flatten()
    
    result_sq = minimize(objective, x0_sq, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    best_points_sq = result_sq.x.reshape(n, 2)
    
    # Compare scores
    def get_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0: return 0
        return (d_min / d_max) ** 2
        
    score_hex = get_score(best_points)
    score_sq = get_score(best_points_sq)
    
    if score_hex >= score_sq:
        return best_points
    else:
        return best_points_sq
