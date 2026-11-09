import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _get_hex_lattice_points(n_points, radius=1.0):
    """
    Generate initial points from a hexagonal lattice clipped to a circle.
    """
    points = []
    # Approximate spacing based on density
    # Area of circle = pi. Area per point ~ pi/n_points.
    # Hexagonal lattice spacing s: area per point ~ s^2 * sqrt(3)/2
    # s ~ sqrt(2 * pi / (n_points * sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # Generate a grid of hexagonal points
    k_max = int(2.0 / s) + 2
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = i * s + 0.5 * s * (j % 2)
            y = j * s * np.sqrt(3) / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter points inside the circle (with a small margin to avoid boundary issues initially)
    distances = np.linalg.norm(points, axis=1)
    # We want points well inside
    mask = distances < radius - s/2
    points = points[mask]
    
    # Sort by distance from origin to pick the central cluster
    if len(points) < n_points:
        # If not enough, expand grid or just take what we have and pad?
        # For N=50, the grid above should be sufficient.
        # Let's ensure we have enough by expanding if necessary
        k_max *= 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = i * s + 0.5 * s * (j % 2)
                y = j * s * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        distances = np.linalg.norm(points, axis=1)
        mask = distances < radius
        points = points[mask]
        
    # Select the n_points closest to the origin
    distances = np.linalg.norm(points, axis=1)
    indices = np.argsort(distances)
    points = points[indices[:n_points]]
    
    return points

def _objective(x):
    """
    Objective function to minimize: negative of the minimum distance.
    x is a flat array of shape (2*N,).
    """
    pts = x.reshape(N, 2)
    
    # Calculate all pairwise distances
    # Using broadcasting for efficiency
    # pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # Distance to boundary
    # dist_to_wall = 1 - norm(pts)
    # We want to maximize min(min_pair_dist, 2*min_dist_to_wall)
    # So we minimize -min(...)
    
    # Get pairwise distances excluding diagonal
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    min_pair_dist = np.min(pair_dists)
    
    # Distance to wall (scaled by 2 to be comparable to pair dist which is 2*r)
    norms = np.linalg.norm(pts, axis=1)
    wall_dists = 2 * (1 - norms)
    min_wall_dist = np.min(wall_dists)
    
    # The effective diameter is the min of the closest pair and the closest wall distance (scaled)
    # Actually, radius r = min(min_pair_dist/2, min(1-norms))
    # Diameter D = 2*r = min(min_pair_dist, 2*min(1-norms))
    # So we maximize D.
    
    current_d = min(min_pair_dist, min_wall_dist)
    
    return -current_d

def _constraints(x):
    """
    Constraints: points must be inside the unit circle.
    """
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    # 1 - norm >= 0
    return [1 - n for n in norms]

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin for 120s limit
    
    # Initial points
    x0 = _get_hex_lattice_points(N)
    x0_flat = x0.flatten()
    
    # Define constraints for SLSQP
    # SLSQP expects a list of dicts or a single dict with 'ineq'
    # Inequality: g(x) >= 0
    cons = {'type': 'ineq', 'fun': lambda x: _constraints(x)}
    
    best_x = x0_flat
    best_score = _objective(best_x)
    
    # Try a few different initializations or restarts if time permits
    # Since N=50, one good run might be enough, but random perturbations help.
    
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        
        # Add small random perturbation to escape local minima
        if attempts > 1:
            x_init = best_x + np.random.normal(0, 0.01, size=2*N)
            # Ensure initial points are valid (inside circle)
            # If not, project them back or just use the previous best
            # For simplicity, we just try to optimize from the perturbed best.
            # SLSQP will handle the constraints.
        else:
            x_init = x0_flat
            
        try:
            res = minimize(_objective, x_init, method='SLSQP', constraints=cons, 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Verify validity
                pts = res.x.reshape(N, 2)
                norms = np.linalg.norm(pts, axis=1)
                if np.all(norms < 1.0):
                    best_x = res.x
                    best_score = res.fun
        except Exception as e:
            pass
            
        # If we've run out of time, break
        if time.time() - start_time > timeout:
            break
            
        # If the improvement is very small, we might stop early
        # But for now, let's just keep trying with perturbations if time allows
        
    return best_x.reshape(N, 2)
