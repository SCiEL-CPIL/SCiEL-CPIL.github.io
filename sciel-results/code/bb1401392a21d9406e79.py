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
    
    def objective(x, t=80.0):
        # x is (30,) -> 15 points * 2 coords
        # z_0 = 0
        z = np.zeros((n, 2))
        z[1:, 0] = x[0::2]
        z[1:, 1] = x[1::2]
        
        d = compute_distances(z)
        if len(d) == 0:
            return 0.0
        
        # Soft min of distances
        d_min_val = np.min(d)
        exp_terms = np.exp(-t * (d - d_min_val))
        soft_min = d_min_val - (1.0/t) * np.log(np.sum(exp_terms))
        
        # Constraint: max distance <= 1
        d_max_val = np.max(d)
        penalty = 100.0 * max(0.0, d_max_val - 1.0)**2
        
        # We want to maximize soft_min, so minimize -soft_min + penalty
        return -soft_min + penalty
    
    best_score = -np.inf
    best_points = None
    
    # Try multiple random restarts with different initializations
    for trial in range(20):
        np.random.seed(42 + trial * 7)
        
        # Mix of random and structured initializations
        if trial < 5:
            # Random points in unit disk
            r = np.sqrt(np.random.rand(15))
            theta = 2 * np.pi * np.random.rand(15)
            x0 = np.zeros(30)
            x0[0::2] = r * np.cos(theta)
            x0[1::2] = r * np.sin(theta)
        elif trial < 10:
            # Points on concentric circles
            x0 = np.zeros(30)
            for i in range(15):
                if i < 6:
                    r = 0.5
                    theta = 2 * np.pi * i / 6
                elif i < 12:
                    r = 0.8
                    theta = 2 * np.pi * (i - 6) / 6 + np.pi / 6
                else:
                    r = 1.0
                    theta = 2 * np.pi * (i - 12) / 3
                x0[2*i] = r * np.cos(theta)
                x0[2*i + 1] = r * np.sin(theta)
        else:
            # Hexagonal lattice perturbation
            x0 = np.zeros(30)
            # Generate hexagonal lattice points
            points_hex = []
            for row in range(-2, 3):
                for col in range(-2, 3):
                    x = col + 0.5 * (row % 2)
                    y = row * np.sqrt(3) / 2
                    points_hex.append((x, y))
            
            # Select 15 points from hexagonal lattice, centered
            points_hex = np.array(points_hex)
            points_hex -= np.mean(points_hex, axis=0)
            # Scale to fit in unit disk
            max_r = np.max(np.sqrt(points_hex[:, 0]**2 + points_hex[:, 1]**2))
            if max_r > 0:
                points_hex = points_hex / max_r * 0.95
            
            # Take first 15 points
            selected = points_hex[:15]
            x0[0::2] = selected[:, 0]
            x0[1::2] = selected[:, 1]
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        # Reconstruct points
        z = np.zeros((n, 2))
        z[1:, 0] = res.x[0::2]
        z[1:, 1] = res.x[1::2]
        
        d = compute_distances(z)
        d_min_val = np.min(d)
        d_max_val = np.max(d)
        
        if d_max_val > 0:
            ratio = (d_min_val / d_max_val) ** 2
        else:
            ratio = 0.0
        
        if ratio > best_score:
            best_score = ratio
            best_points = z.copy()
    
    # Local refinement: try small perturbations around the best solution
    if best_points is not None:
        for _ in range(5):
            # Add small random perturbation
            noise = 0.01 * np.random.randn(30)
            x_perturbed = np.zeros(30)
            x_perturbed[0::2] = best_points[1:, 0] + noise[0::2]
            x_perturbed[1::2] = best_points[1:, 1] + noise[1::2]
            
            res = minimize(objective, x_perturbed, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
            
            z = np.zeros((n, 2))
            z[1:, 0] = res.x[0::2]
            z[1:, 1] = res.x[1::2]
            
            d = compute_distances(z)
            d_min_val = np.min(d)
            d_max_val = np.max(d)
            
            if d_max_val > 0:
                ratio = (d_min_val / d_max_val) ** 2
            else:
                ratio = 0.0
            
            if ratio > best_score:
                best_score = ratio
                best_points = z.copy()
    
    # Rescale to have d_max = 1
    if best_points is not None:
        d = compute_distances(best_points)
        d_max_val = np.max(d)
        if d_max_val > 0:
            best_points = best_points / d_max_val
    
    return best_points


# EVOLVE-BLOCK-END
