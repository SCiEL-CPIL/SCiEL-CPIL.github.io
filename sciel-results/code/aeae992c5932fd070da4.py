import numpy as np
import time
import sys

N = 50

def _hex_lattice_init():
    """Generate initial points on a hexagonal lattice clipped to unit circle."""
    # Estimate spacing
    # Area of unit circle = pi. Area per circle ~ pi/N.
    # Spacing s ~ sqrt(pi/N)
    s = np.sqrt(np.pi / N)
    
    points = []
    # Generate grid
    # Hex lattice:
    # x = i * s + 0.5 * (j % 2) * s
    # y = j * s * sqrt(3) / 2
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = i * s + 0.5 * (j % 2) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 1.0 - s/2.0: # Ensure inside
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback to random
        return np.random.uniform(-0.8, 0.8, (N, 2))
    
    # Select N points closest to center to start dense
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii)[:N]
    return points[idx]

def _compute_radius_and_grad(points):
    """
    Compute the maximum radius r and the gradient of r w.r.t points.
    points: (N, 2) array
    Returns: r, grad (N, 2) array
    """
    n = points.shape[0]
    # Compute pairwise distances
    # dist[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Wall distances
    wall_dist = 1.0 - np.linalg.norm(points, axis=1)
    
    r_pairs = np.min(dist / 2.0)
    r_walls = np.min(wall_dist)
    
    r = min(r_pairs, r_walls)
    
    # Gradient calculation
    grad = np.zeros_like(points)
    
    # Tolerance for active constraints
    tol = 1e-5
    
    # Check if pair constraints are active
    if r_pairs <= r_walls + tol:
        # Find all pairs with dist/2 close to r_pairs
        # We want to push points apart
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] / 2.0 <= r_pairs + tol:
                    d = diff[i, j]
                    norm_d = np.linalg.norm(d)
                    if norm_d > 1e-9:
                        u = d / norm_d
                        grad[i] += u
                        grad[j] -= u
    
    # Check if wall constraints are active
    if r_walls <= r_pairs + tol:
        for i in range(n):
            if wall_dist[i] <= r_walls + tol:
                norm_p = np.linalg.norm(points[i])
                if norm_p > 1e-9:
                    # Push towards center
                    u = -points[i] / norm_p
                    grad[i] += u
    
    return r, grad

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s buffer
    
    best_r = -1.0
    best_points = None
    
    # Initial best from hex lattice
    init_points = _hex_lattice_init()
    r, _ = _compute_radius_and_grad(init_points)
    best_r = r
    best_points = init_points.copy()
    
    # Optimization loop
    # We will try multiple restarts with perturbations
    # and gradient ascent
    
    max_restarts = 50
    steps_per_restart = 200
    
    rng = np.random.default_rng(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize current points
        if restart == 0:
            curr_points = init_points.copy()
        else:
            # Perturb the best found so far or start fresh
            if rng.random() < 0.5:
                curr_points = best_points.copy()
                # Add small noise
                curr_points += rng.normal(0, 0.01, curr_points.shape)
            else:
                curr_points = _hex_lattice_init()
                # Shuffle and select? Or just use fresh lattice
                # Add some randomness to lattice selection
                # For now, just perturb fresh lattice
                curr_points += rng.normal(0, 0.05, curr_points.shape)
        
        # Ensure points are inside circle
        radii = np.linalg.norm(curr_points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            curr_points[mask] *= 0.99 / radii[mask, np.newaxis]
            
        curr_r, _ = _compute_radius_and_grad(curr_points)
        
        # Gradient Ascent
        step_size = 0.01
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            curr_r, grad = _compute_radius_and_grad(curr_points)
            
            if curr_r > best_r:
                best_r = curr_r
                best_points = curr_points.copy()
                
            # Normalize gradient to prevent large jumps
            grad_norms = np.linalg.norm(grad, axis=1)
            # Avoid division by zero
            grad_norms[grad_norms < 1e-9] = 1.0
            grad = grad / grad_norms[:, np.newaxis]
            
            # Update
            curr_points += step_size * grad
            
            # Project back to unit circle if needed
            radii = np.linalg.norm(curr_points, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                curr_points[mask] *= 0.99 / radii[mask, np.newaxis]
                
            # Adaptive step size
            new_r, _ = _compute_radius_and_grad(curr_points)
            if new_r > curr_r:
                step_size *= 1.05
            else:
                step_size *= 0.9
                
            if step_size < 1e-6:
                break
                
    # Final verification
    # Ensure the returned points are valid
    final_r, _ = _compute_radius_and_grad(best_points)
    if final_r < 0:
        # Fallback to hex lattice if something went wrong
        best_points = _hex_lattice_init()
        
    return best_points
