# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    
    def compute_distances(points):
        return pdist(points)
    
    def build_points_from_params(params):
        """Build 16 points from 30 parameters: 15 angles + 15 radii.
        Point 0 is fixed at origin. Points 1-15 have polar coordinates (r_i, theta_i)."""
        z = np.zeros((n, 2))
        # z[0] = (0, 0)
        for i in range(15):
            r = params[2*i]
            theta = params[2*i + 1]
            z[i+1, 0] = r * np.cos(theta)
            z[i+1, 1] = r * np.sin(theta)
        return z
    
    def objective(params):
        """Objective to minimize: -log-sum-exp surrogate for min distance + penalty for max distance > 1"""
        z = build_points_from_params(params)
        d = compute_distances(z)
        
        if len(d) == 0:
            return 0.0
        
        # Soft min of distances using log-sum-exp
        t = 50.0  # temperature parameter for soft min
        d_min_val = np.min(d)
        # Use stable log-sum-exp
        shifted = d - d_min_val
        exp_terms = np.exp(-t * shifted)
        soft_min = d_min_val - (1.0/t) * np.log(np.sum(exp_terms))
        
        # Constraint: max distance <= 1
        d_max_val = np.max(d)
        penalty = 1000.0 * max(0.0, d_max_val - 1.0)**2
        
        # We want to maximize soft_min, so minimize -soft_min + penalty
        return -soft_min + penalty
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Try multiple random restarts with different initializations
    n_trials = 30
    
    for trial in range(n_trials):
        np.random.seed(42 + trial * 13)
        
        if trial < 10:
            # Random points in unit disk with polar coordinates
            radii = np.sqrt(np.random.rand(15)) * 0.95
            angles = 2 * np.pi * np.random.rand(15)
            params = np.zeros(30)
            params[0::2] = radii
            params[1::2] = angles
            
        elif trial < 20:
            # Structured: concentric rings
            params = np.zeros(30)
            # 6 points on inner ring
            for i in range(6):
                params[2*i] = 0.5
                params[2*i + 1] = 2 * np.pi * i / 6
            # 6 points on middle ring
            for i in range(6):
                params[2*(i+6)] = 0.8
                params[2*(i+6) + 1] = 2 * np.pi * (i+6) / 6 + np.pi / 6
            # 3 points on outer ring
            for i in range(3):
                params[2*(i+12)] = 1.0
                params[2*(i+12) + 1] = 2 * np.pi * (i+12) / 6
            
        else:
            # Hexagonal lattice-like structure
            params = np.zeros(30)
            # Place points in a hexagonal pattern
            hex_points = []
            # Center is at origin (point 0)
            # Ring 1: 6 points at radius ~0.5
            for i in range(6):
                theta = 2 * np.pi * i / 6
                hex_points.append((0.5, theta))
            # Ring 2: 6 points at radius ~0.85
            for i in range(6):
                theta = 2 * np.pi * i / 6 + np.pi / 6
                hex_points.append((0.85, theta))
            # Ring 3: 3 points at radius ~1.0
            for i in range(3):
                theta = 2 * np.pi * i / 6
                hex_points.append((1.0, theta))
            
            for i, (r, theta) in enumerate(hex_points):
                params[2*i] = r
                params[2*i + 1] = theta
        
        # Optimize using L-BFGS-B
        res = minimize(
            objective, 
            params, 
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        # Reconstruct points
        z = build_points_from_params(res.x)
        
        d = compute_distances(z)
        d_min_val = np.min(d)
        d_max_val = np.max(d)
        
        if d_max_val > 0:
            ratio = (d_min_val / d_max_val) ** 2
        else:
            ratio = 0.0
        
        if ratio > best_score:
            best_score = ratio
            best_params = res.x.copy()
            best_points = z.copy()
    
    # Local refinement: try small perturbations around the best solution
    if best_params is not None:
        for _ in range(10):
            # Add small random perturbation
            noise = 0.05 * np.random.randn(30)
            params_perturbed = best_params + noise
            
            res = minimize(
                objective, 
                params_perturbed, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            z = build_points_from_params(res.x)
            
            d = compute_distances(z)
            d_min_val = np.min(d)
            d_max_val = np.max(d)
            
            if d_max_val > 0:
                ratio = (d_min_val / d_max_val) ** 2
            else:
                ratio = 0.0
            
            if ratio > best_score:
                best_score = ratio
                best_params = res.x.copy()
                best_points = z.copy()
    
    # Rescale to have d_max = 1
    if best_points is not None:
        d = compute_distances(best_points)
        d_max_val = np.max(d)
        if d_max_val > 0:
            best_points = best_points / d_max_val
    
    return best_points


# EVOLVE-BLOCK-END
