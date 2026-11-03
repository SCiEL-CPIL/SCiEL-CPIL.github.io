import numpy as np
import time
from scipy.optimize import minimize

N = 50

def get_radius(centers):
    """
    Calculate the maximum radius for given centers.
    centers: (N, 2) array
    """
    # Pairwise distances
    # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - radii
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def objective(x):
    """
    Objective function for scipy.optimize.minimize.
    x: (2N,) array
    Returns negative radius to maximize radius.
    """
    centers = x.reshape(N, 2)
    return -get_radius(centers)

def generate_hexagonal_init(jitter=0.0):
    """
    Generate an initial hexagonal lattice configuration.
    """
    # Approximate spacing for 50 circles in unit circle
    # Area of unit circle = pi. Area per circle approx pi/50.
    # Circle area = pi * r^2. r ~ sqrt(1/50) ~ 0.14.
    # Spacing ~ 2*r ~ 0.28.
    
    # We'll generate a grid and clip
    spacing = 0.25
    points = []
    # Generate grid points
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing
            y = j * spacing * np.sqrt(3)
            if i % 2 == 1:
                y += spacing * np.sqrt(3) / 2
            # Check if inside unit circle with some margin
            if np.sqrt(x*x + y*y) < 0.9:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have too many, randomly select N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points if necessary (should not happen with this grid size)
        while len(points) < N:
            p = np.random.uniform(-0.9, 0.9, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])
                
    # Add jitter
    if jitter > 0:
        points += np.random.normal(0, jitter, points.shape)
        
    # Ensure points are inside
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 1.0
    if np.sum(mask) < N:
        # Re-randomize any points outside
        for i in range(len(points)):
            if radii[i] >= 1.0:
                while True:
                    p = np.random.uniform(-1, 1, 2)
                    if np.linalg.norm(p) < 1.0:
                        points[i] = p
                        break
                        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_r = 0.0
    best_centers = None
    
    # Try multiple restarts
    # Start with hexagonal lattice + jitter
    # Then try pure random
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Alternate between hex and random initialization
        if num_restarts % 3 == 0:
            # Random initialization
            centers = np.random.uniform(-1, 1, (N, 2))
            radii = np.linalg.norm(centers, axis=1)
            mask = radii < 1.0
            # Resample points outside
            while np.sum(mask) < N:
                idx = np.where(~mask)[0]
                for i in idx:
                    while True:
                        p = np.random.uniform(-1, 1, 2)
                        if np.linalg.norm(p) < 1.0:
                            centers[i] = p
                            mask[i] = True
                            break
        else:
            # Hexagonal with jitter
            jitter = 0.05 if num_restarts > 5 else 0.01
            centers = generate_hexagonal_init(jitter=jitter)
        
        # Flatten for optimizer
        x0 = centers.flatten()
        
        # Optimize
        # Powell is robust for this non-smooth objective
        try:
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            x_opt = res.x
            r_opt = -res.fun
        except Exception:
            x_opt = x0
            r_opt = -objective(x0)
            
        if r_opt > best_r:
            best_r = r_opt
            best_centers = x_opt.reshape(N, 2)
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final check to ensure validity
    if best_centers is None:
        # Fallback to a simple valid configuration
        best_centers = generate_hexagonal_init(jitter=0.0)
        
    return best_centers
