import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _initialize_hexagonal(N):
    """Initialize centers on a hexagonal lattice."""
    # Estimate side length of square in terms of lattice spacing
    # Area of hex lattice unit cell is sqrt(3)/2 * s^2
    # N * (sqrt(3)/2 * s^2) ~ 1
    # s ~ sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate hex lattice points
    # We need enough points to cover the square [-0.5, 0.5]^2
    # Range in x: -0.5 to 0.5 -> width 1. Number of columns ~ 1/s
    # Range in y: -0.5 to 0.5 -> height 1. Number of rows ~ 1/(s*sqrt(3)/2)
    
    n_cols = int(np.ceil(1.0 / s)) + 2
    n_rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    
    # Start from a negative offset to center the lattice
    x_start = -0.5 - s * n_cols // 2
    y_start = -0.5 - (s * np.sqrt(3) / 2) * n_rows // 2
    
    points = []
    for row in range(n_rows):
        y = y_start + row * s * np.sqrt(3) / 2
        # Offset even/odd rows
        offset = 0 if row % 2 == 0 else s / 2
        for col in range(n_cols):
            x = x_start + col * s + offset
            # Check if point is roughly inside or near the square
            # We'll filter later, but let's generate a grid around the square
            if -0.6 <= x <= 0.6 and -0.6 <= y <= 0.6:
                points.append((x, y))
    
    points = np.array(points)
    
    # We might have more or less than N points.
    # If more, select the ones closest to the center or just take the first N?
    # Better: select N points that are well-distributed.
    # Simple approach: take the points closest to the origin? No, that clumps them.
    # The hex lattice is regular. Let's just take the first N points in a sorted order?
    # Actually, the generation above creates a grid.
    # Let's ensure we have at least N points.
    if len(points) < N:
        # Increase density or range
        s *= 0.9
        return _initialize_hexagonal(N)
        
    # Select N points. 
    # To maximize packing, we want to keep the "core" of the lattice.
    # Points far from the center might be cut off by the square boundary anyway.
    # Let's sort by distance from center and take the closest N? 
    # This might bias towards center, which is good for packing density in a square.
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    selected = points[indices[:N]]
    
    # Shuffle to avoid any ordering bias in optimization? 
    # Not strictly necessary for L-BFGS-B, but good practice.
    np.random.shuffle(selected)
    
    return selected

def _soft_min_energy(c, lam=50.0):
    """
    Compute soft-min energy.
    Minimizing this approximates maximizing the minimum distance.
    E = sum_{i<j} exp(-lam * d_ij) + sum_i exp(-lam * d_boundary_i)
    """
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle to avoid double counting, but summing all is fine
    # Just be careful with the inf on diagonal.
    # Mask diagonal
    mask = ~np.eye(len(c), dtype=bool)
    d_pairs = dists[mask]
    
    # Boundary distances
    # Distance to each of the 4 boundaries
    db_x1 = 0.5 - c[:, 0]  # distance to x = 0.5
    db_x2 = 0.5 + c[:, 0]  # distance to x = -0.5
    db_y1 = 0.5 - c[:, 1]  # distance to y = 0.5
    db_y2 = 0.5 + c[:, 1]  # distance to y = -0.5
    
    d_bounds = np.min(np.stack([db_x1, db_x2, db_y1, db_y2]), axis=1)
    
    # Soft sum
    # To avoid overflow, we can shift by min, but exp(-lam * d) for small d is large.
    # We want to minimize the sum. Large values for small distances.
    # This is a penalty.
    
    # Clip distances to avoid extreme values if points overlap heavily
    d_pairs_clipped = np.clip(d_pairs, 1e-6, None)
    d_bounds_clipped = np.clip(d_bounds, 1e-6, None)
    
    e_pairs = np.sum(np.exp(-lam * d_pairs_clipped))
    e_bounds = np.sum(np.exp(-lam * d_bounds_clipped))
    
    return e_pairs + e_bounds

def _compute_radius(c):
    """Compute the exact valid radius."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    dist_to_bound = np.min(0.5 - np.abs(c), axis=1)
    min_bound_dist = np.min(dist_to_bound)
    
    r = min(min_pair_dist / 2.0, min_bound_dist)
    return r if r > 0 else 1e-15

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    # 1. Initialization
    centers = _initialize_hexagonal(N)
    
    # 2. Soft-Min Optimization with L-BFGS-B
    # We define the objective as the soft energy.
    # We use bounds to keep points inside the square (with a small margin for numerical stability)
    # However, the soft boundary penalty handles the "soft" constraint.
    # Hard bounds in L-BFGS-B can sometimes cause issues if the gradient points outwards.
    # Let's use bounds [-0.499, 0.499] to keep them strictly inside.
    
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try a few different lambda values for the soft-min
    # Smaller lambda = smoother, easier to optimize, but less accurate approximation.
    # Larger lambda = sharper, closer to true min, but harder to optimize.
    lambdas = [20.0, 50.0, 100.0]
    
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
            
        def obj(c):
            return _soft_min_energy(c, lam)
            
        try:
            result = minimize(
                obj,
                best_centers,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < obj(best_centers):
                new_radius = _compute_radius(result.x)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = result.x.copy()
        except Exception as e:
            pass
            
    # 3. Final Polish with Nelder-Mead on the exact objective
    # The exact objective is non-smooth. Nelder-Mead is derivative-free and handles this well.
    if time.time() < deadline - 5:
        def exact_obj(c):
            return -_compute_radius(c)
            
        try:
            result = minimize(
                exact_obj,
                best_centers,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < exact_obj(best_centers):
                best_centers = result.x.copy()
        except Exception:
            pass
            
    # 4. Final Clip to ensure validity
    final_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return final_centers
