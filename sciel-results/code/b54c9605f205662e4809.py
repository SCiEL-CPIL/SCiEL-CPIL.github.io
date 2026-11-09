import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    
    def objective(params):
        # params: shape (n, 2) -> theta, phi
        theta = params[:, 0]
        phi = params[:, 1]
        
        # Convert to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.column_stack((x, y, z))
        
        # Calculate pairwise distances
        dists = pdist(points, metric='euclidean')
        
        if len(dists) == 0:
            return 0.0
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return -(ratio ** 2)
    
    def params_to_points(params):
        theta = params[:, 0]
        phi = params[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))

    best_points = None
    best_score = -np.inf

    # Multiple starts
    n_starts = 40
    
    for i in range(n_starts):
        if i == 0:
            # Start 0: Fibonacci sphere
            z = 1 - 2 * np.arange(n) / (n - 1)
            r = np.sqrt(1 - z**2)
            phi = np.arange(n) * np.pi * (3. - np.sqrt(5.))
            theta = np.arccos(z)
            init_params = np.column_stack((theta, phi))
        else:
            # Random starts
            np.random.seed(i * 100 + 42)
            theta = np.random.uniform(0, np.pi, n)
            phi = np.random.uniform(0, 2 * np.pi, n)
            init_params = np.column_stack((theta, phi))
            
        # Nelder-Mead optimization
        res = minimize(
            objective, 
            init_params, 
            method='Nelder-Mead', 
            options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-8}
        )
        
        current_points = params_to_points(res.x)
        # Calculate actual score on the points
        dists = pdist(current_points, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max > 1e-10:
            score = (d_min / d_max) ** 2
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_points = current_points

    return best_points
