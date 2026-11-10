# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a 4x4 lattice with radial distortion (power-law or rational).
    """
    n = 16
    d = 2

    # Create base 4x4 grid centered at origin
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    base_points = np.vstack((xx.ravel(), yy.ravel())).T
    
    # Calculate radii
    radii = np.linalg.norm(base_points, axis=1)
    
    best_R = -1
    best_points = base_points
    
    # Type 1: Power Law r' = r^alpha
    # Scan alpha
    alphas = np.linspace(0.3, 1.5, 500)
    for alpha in alphas:
        new_radii = radii ** alpha
        scale_factors = new_radii / radii
        warped_points = base_points * scale_factors[:, np.newaxis]
        
        dists = pdist(warped_points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            R = 0
        else:
            R = (d_min / d_max) ** 2
            
        if R > best_R:
            best_R = R
            best_points = warped_points
            
    # Refine Power Law
    # Find the alpha that gave best_R? We didn't store it. Let's just re-scan fine.
    # Actually, let's store best params.
    
    # Let's do a proper loop with tracking.
    
    best_R = -1
    best_points = base_points
    best_type = "power"
    best_param = 1.0
    
    # 1. Power Law
    for alpha in np.linspace(0.3, 1.5, 1000):
        new_radii = radii ** alpha
        scale_factors = new_radii / radii
        warped_points = base_points * scale_factors[:, np.newaxis]
        
        dists = pdist(warped_points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        R = (d_min / d_max) ** 2 if d_max > 0 else 0
        
        if R > best_R:
            best_R = R
            best_points = warped_points
            best_type = "power"
            best_param = alpha

    # 2. Rational Warp r' = r / (1 + k * r^2)
    # k > 0 compresses outer points.
    # If k is large, r' ~ 1/(k r).
    for k in np.linspace(0.0, 1.0, 1000):
        # Handle k=0 -> identity
        if k == 0:
            new_radii = radii
        else:
            new_radii = radii / (1 + k * radii**2)
            
        scale_factors = new_radii / radii
        warped_points = base_points * scale_factors[:, np.newaxis]
        
        dists = pdist(warped_points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        R = (d_min / d_max) ** 2 if d_max > 0 else 0
        
        if R > best_R:
            best_R = R
            best_points = warped_points
            best_type = "rational"
            best_param = k

    # Refine around best_param
    if best_type == "power":
        range_val = 0.05
    else:
        range_val = 0.05
        
    if best_type == "power":
        params = np.linspace(best_param - range_val, best_param + range_val, 1000)
        for alpha in params:
            new_radii = radii ** alpha
            scale_factors = new_radii / radii
            warped_points = base_points * scale_factors[:, np.newaxis]
            
            dists = pdist(warped_points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            R = (d_min / d_max) ** 2 if d_max > 0 else 0
            
            if R > best_R:
                best_R = R
                best_points = warped_points
    else:
        params = np.linspace(best_param - range_val, best_param + range_val, 1000)
        for k in params:
            if k < 0: continue
            if k == 0:
                new_radii = radii
            else:
                new_radii = radii / (1 + k * radii**2)
                
            scale_factors = new_radii / radii
            warped_points = base_points * scale_factors[:, np.newaxis]
            
            dists = pdist(warped_points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            R = (d_min / d_max) ** 2 if d_max > 0 else 0
            
            if R > best_R:
                best_R = R
                best_points = warped_points

    return best_points

# EVOLVE-BLOCK-END
