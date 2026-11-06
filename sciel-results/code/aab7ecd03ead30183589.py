"""
Structured Lattice Packing Strategy for 50 circles in a unit square.

Strategy:
1. Define a parametric family of hexagonal lattice packings.
   - Parameters: dx (horizontal spacing), dy (vertical spacing), rotation, 
     x_offset, y_offset, and row-wise adjustments.
2. Use Nelder-Mead to optimize these parameters to maximize the minimum radius.
   This is a lower-dimensional optimization problem (fewer than 10 variables 
   vs 100) and avoids the pitfalls of optimizing 100 individual coordinates 
   from a poor start.
3. Take the best lattice configuration and perform a final local search 
   (individual coordinate optimization) to break lattice symmetry and fine-tune.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)

def _generate_hex_lattice(params):
    """
    Generate a hexagonal lattice configuration from parameters.
    params: [dx, dy, rotation, x_off, y_off, row_shift_amp, row_shift_freq, edge_squeeze]
    
    We construct rows of circles. The number of circles per row is determined 
    by fitting them within the width. To ensure exactly N=50 circles, we 
    dynamically adjust the number of rows and circles per row to sum to N.
    
    A more robust approach for fixed N:
    Pre-define a set of "row configurations" (lists of counts per row) that sum to N.
    Then for each configuration, generate the lattice points and optimize.
    """
    dx, dy, rot, x_off, y_off, row_shift_amp, row_shift_freq, edge_squeeze = params
    
    # We will try to construct a lattice that roughly fits N points.
    # Let's use a standard hexagonal lattice generation and then trim/pad to N.
    # Actually, for a specific N, it's better to define the rows explicitly.
    
    # Let's define a few candidate row structures for 50 circles.
    # Example: 7, 6, 7, 6, 7, 6, 7, 4 (Sum=50)
    # Or: 8, 7, 8, 7, 8, 7, 5 (Sum=50)
    # Or: 10, 9, 10, 9, 10, 2 (Sum=50) -- unlikely to be optimal geometry
    
    # To keep it simple and effective, we will generate a dense hex lattice 
    # centered at (0,0) and select the N closest points to the center.
    # Then we apply the transformations (rotation, scaling via dx/dy, translation).
    # This ensures we always have exactly N points in a roughly circular/elliptical 
    # hexagonal cluster, which we then fit into the square.
    
    # Generate points in a hex lattice
    max_radius = 10 # Enough to cover 50 points
    points = []
    for i in range(-max_radius, max_radius + 1):
        for j in range(-max_radius, max_radius + 1):
            # Hex lattice coordinates to Cartesian
            # Standard hex grid: 
            # x = (i + j/2) * sqrt(3)
            # y = j * 1.5
            x = (i + j / 2.0) * np.sqrt(3)
            y = j * 1.5
            points.append([x, y])
            
    points = np.array(points)
    
    # Keep only points within a certain radius to have a manageable set
    dists_from_origin = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    # Sort by distance from origin and take the first N
    sorted_indices = np.argsort(dists_from_origin)
    selected_points = points[sorted_indices[:N]]
    
    # Now apply scaling and rotation
    # dx and dy control the aspect ratio and overall scale
    # We want to scale such that the bounding box fits in [-0.5, 0.5]
    
    # Normalize the selected points to have unit max norm (roughly)
    max_norm = np.max(np.linalg.norm(selected_points, axis=1))
    if max_norm > 0:
        selected_points = selected_points / max_norm
    
    # Apply anisotropic scaling
    selected_points[:, 0] *= dx
    selected_points[:, 1] *= dy
    
    # Apply rotation
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    selected_points = selected_points @ rot_matrix.T
    
    # Apply translation
    selected_points[:, 0] += x_off
    selected_points[:, 1] += y_off
    
    # Edge squeeze: pull points near the boundary inward slightly to account for 
    # the square shape vs circular lattice. This is a heuristic.
    # If a point is close to the edge, move it slightly towards the center.
    # This helps satisfy boundary constraints.
    for i in range(N):
        for k in range(2):
            val = selected_points[i, k]
            if val > 0.4: # Close to right edge
                selected_points[i, k] -= (val - 0.4) * edge_squeeze * 0.1
            elif val < -0.4: # Close to left edge
                selected_points[i, k] -= (val + 0.4) * edge_squeeze * 0.1
            elif val > 0.4: # Close to top edge
                selected_points[i, k] -= (val - 0.4) * edge_squeeze * 0.1
            elif val < -0.4: # Close to bottom edge
                selected_points[i, k] -= (val + 0.4) * edge_squeeze * 0.1
                
    return selected_points

def _lattice_objective(params):
    """Objective function for lattice optimization: negative radius."""
    centers = _generate_hex_lattice(params)
    # Check validity
    if np.any(np.abs(centers) > 0.5 + 1e-6):
        return 10.0 # Large penalty
    r = _fast_radius(centers)
    return -r

def _optimize_lattice(initial_params, deadline):
    """Optimize lattice parameters using Nelder-Mead."""
    start_time = time.time()
    
    def obj(x):
        if time.time() > deadline:
            return 1e10
        return _lattice_objective(x)
    
    try:
        res = minimize(
            obj, 
            initial_params, 
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        best_params = res.x
        best_centers = _generate_hex_lattice(best_params)
        best_r = _fast_radius(best_centers)
        return best_centers, best_r, best_params
    except Exception as e:
        # Fallback to initial
        centers = _generate_hex_lattice(initial_params)
        r = _fast_radius(centers)
        return centers, r, initial_params

def _local_search(centers, n_iterations=2000, deadline=None):
    """Local search on individual coordinates to polish the lattice result."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    rng = np.random.RandomState(42)
    
    current_step = 0.02
    no_improve_count = 0
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.randint(N)
        orig = best_centers[idx].copy()
        
        improved = False
        for attempt in range(3):
            s = current_step * (0.5 ** attempt)
            perturbation = rng.normal(0, s, size=2)
            candidate = orig + perturbation
            
            np.clip(candidate, -0.5 + 1e-8, 0.5 - 1e-8, out=candidate)
            best_centers[idx] = candidate
            
            new_radius = _fast_radius(best_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                improved = True
                break
            else:
                best_centers[idx] = orig
        
        if improved:
            no_improve_count = 0
            current_step = min(0.05, current_step * 1.02)
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                current_step = max(0.001, current_step * 0.8)
                no_improve_count = 0
                
    return best_centers, best_radius

def construct():
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Define several initial parameter guesses for the lattice
    # [dx, dy, rotation, x_off, y_off, row_shift_amp, row_shift_freq, edge_squeeze]
    # dx, dy should be around 0.1-0.2 for 50 circles in a unit square
    initial_guesses = [
        [0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.18, 0.18, np.pi/6, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.12, 0.12, np.pi/12, 0.0, 0.0, 0.0, 0.0, 0.5],
        [0.16, 0.14, 0.1, 0.01, -0.01, 0.0, 0.0, 0.2],
        [0.14, 0.16, -0.1, -0.01, 0.01, 0.0, 0.0, 0.2],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Allocate time to lattice optimization
        # Nelder-Mead is relatively fast, give it a chunk
        lattice_deadline = min(deadline, time.time() + remaining * 0.6)
        
        centers, r, params = _optimize_lattice(guess, lattice_deadline)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # If we have a good lattice solution, polish it with local search
    if best_centers is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 10.0
        # Scale iterations based on time
        n_iters = max(500, int(5000 * remaining / 60.0))
        best_centers, best_radius = _local_search(best_centers, n_iterations=n_iters, deadline=deadline)
        
    # Fallback: if lattice didn't work well, try a simple grid + local search
    if best_radius < 0.05: # Very low radius, likely failed
        # Simple grid init
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(float)
        best_radius = _compute_radius(best_centers)
        
        if time.time() < deadline - 5.0:
            remaining = deadline - time.time() - 5.0
            n_iters = max(200, int(2000 * remaining / 30.0))
            best_centers, best_radius = _local_search(best_centers, n_iterations=n_iters, deadline=deadline)
            
    # Final safety check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0 or not np.all(np.isfinite(best_centers)):
        # Last resort: tight grid
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(float)
        
    return best_centers

# EVOLVE-BLOCK-END
