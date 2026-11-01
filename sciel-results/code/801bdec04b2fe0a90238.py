"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hexagonal_lattice(n):
    """Generate a dense hexagonal lattice arrangement of n points in [-0.5, 0.5]^2."""
    # Start with a hexagonal packing pattern
    # For n points, we want to arrange them in a hexagonal pattern
    # The optimal hexagonal packing has spacing d = 1/sqrt(3) * something
    
    # Let's create a hexagonal grid
    # In a hexagonal lattice, points are at:
    # (i * d, j * d * sqrt(3) + (i % 2) * d * sqrt(3)/2)
    
    # We need to fit n points in [-0.5, 0.5]^2
    # Let's try different spacings and find the best fit
    
    best_points = None
    best_count = 0
    
    # Try various spacings
    for spacing in np.linspace(0.01, 0.5, 50):
        # Generate hexagonal lattice points
        points = []
        # x ranges from -0.5 to 0.5
        # y ranges from -0.5 to 0.5
        
        # Hexagonal lattice:
        # Row j: x = i * spacing + (j % 2) * spacing/2, y = j * spacing * sqrt(3)/2
        # Center the lattice
        
        max_j = int(1.0 / (spacing * np.sqrt(3) / 2)) + 2
        max_i = int(1.0 / spacing) + 2
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n and len(points) > best_count:
            best_count = len(points)
            best_points = points
    
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    # If we have more than n points, select the best n
    if len(best_points) > n:
        # Use a greedy approach or just take the first n
        # Better: use k-means style selection or just random sample
        # For simplicity, take a well-distributed subset
        # Sort by distance from center and pick closest n? No, we want spread.
        # Just take every k-th point
        idx = np.linspace(0, len(best_points) - 1, n).astype(int)
        best_points = best_points[idx]
    
    return best_points[:n]


def _distance_objective(x, n):
    """Objective: negative of the minimum pairwise distance (to maximize min distance)."""
    points = x.reshape(-1, 2)
    # Compute all pairwise distances
    # Use scipy's cdist or manual
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Also consider distance to boundary
    # Distance from point to boundary of [-0.5, 0.5]^2
    # For a point (x, y), the distance to the nearest boundary is:
    # min(0.5 - |x|, 0.5 - |y|)
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    # The radius is limited by both min pairwise distance / 2 and min boundary distance
    # But for the objective, we want to maximize the minimum of these
    # Actually, the score is min(min_pairwise_dist / 2, min_boundary_dist)
    # So we want to maximize min(min_pairwise_dist / 2, min_boundary_dist)
    
    return -min(min_dist / 2, min_boundary)


def _project(x, n):
    """Project points back into the unit square [-0.5, 0.5]^2."""
    points = x.reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    return points.flatten()


def construct():
    # Start with hexagonal lattice warm-start
    x0 = _hexagonal_lattice(N).flatten()
    
    # Use projected gradient descent to maximize the minimum distance
    # We'll use scipy's minimize with a custom approach
    
    # Actually, let's use a simpler iterative approach:
    # 1. Start with hexagonal lattice
    # 2. Iteratively adjust points to increase minimum distance
    
    points = x0.reshape(-1, 2).copy()
    
    # Iterative optimization
    for iteration in range(500):
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair
        min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_dist_idx
        
        # Push these two points apart
        direction = points[i] - points[j]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0])
        
        # Move them apart slightly
        step = 0.01
        points[i] += direction * step / 2
        points[j] -= direction * step / 2
        
        # Also push points away from boundaries if too close
        for k in range(N):
            for dim in range(2):
                if points[k, dim] > 0.45:
                    points[k, dim] -= 0.005
                elif points[k, dim] < -0.45:
                    points[k, dim] += 0.005
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    # Final refinement using scipy minimize
    # Define the objective as negative of the score
    def objective(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        return -min(min_pair / 2, min_boundary)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    result = minimize(objective, points.flatten(), method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success or result.fun < objective(points.flatten()):
        points = result.x.reshape(-1, 2)
    
    # Ensure points are within bounds
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    return points
# EVOLVE-BLOCK-END
