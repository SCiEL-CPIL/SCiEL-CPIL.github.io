import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Direct construction of the optimal circular arc path
    # Based on knowledge: circular arc centered at (-1, -1) yields ~1.246
    
    # Number of points to return. 
    # The evaluator adds (0,0,0) and (-2,0,90).
    # The previous code returned 91 points (a=0..90).
    # Let's return 91 points to maintain similar resolution.
    n_points = 91
    a_vals = np.linspace(0, 90, n_points)
    
    # Center of the circular arc
    cx = -1.0
    cy = -1.0
    radius = np.sqrt(2.0)
    
    # Theta starts at 45 degrees (pi/4) and ends at 135 degrees (3*pi/4)
    # a is in degrees.
    # theta(a) = 45 + a (degrees)
    theta_deg = 45.0 + a_vals
    theta_rad = np.deg2rad(theta_deg)
    
    xs = cx + radius * np.cos(theta_rad)
    ys = cy + radius * np.sin(theta_rad)
    
    path_pts = np.column_stack((xs, ys, a_vals))
    
    # Ensure finite
    if not np.all(np.isfinite(path_pts)):
        # Fallback to linear path if something went wrong (should not happen)
        xs = np.linspace(0, -2, n_points)
        ys = np.zeros(n_points)
        path_pts = np.column_stack((xs, ys, a_vals))
        
    return path_pts
