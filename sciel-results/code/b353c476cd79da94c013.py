"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist


def _hexagonal_lattice(n):
    """Generate a dense hexagonal lattice arrangement of n points in [-0.5, 0.5]^2."""
    best_points = None
    best_spacing = 0
    
    # Try various spacings to find one that fits at least n points
    for spacing in np.linspace(0.01, 0.1, 100):
        points = []
        max_j = int(1.0 / (spacing * np.sqrt(3) / 2)) + 2
        max_i = int(1.0 / spacing) + 2
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n and spacing > best_spacing:
            best_spacing = spacing
            best_points = points
    
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    # If we have more than n points, select a well-distributed subset
    if len(best_points) > n:
        # Use a greedy farthest-point sampling
        selected = [best_points[0]]
        remaining = list(range(1, len(best_points)))
        
        for _ in range(n - 1):
            min_dists = np.min(cdist(np.array(selected), best_points[remaining]), axis=0)
            idx = np.argmax(min_dists)
            selected.append(best_points[remaining[idx]])
            remaining.pop(idx)
        
        best_points = np.array(selected)
    
    return best_points[:n]


def _compute_score(points):
    """Compute the radius score: min(min_pairwise_dist / 2, min_boundary_dist)."""
    if len(points) < 2:
        return 0.5
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2, min_boundary)


def construct():
    # Start with hexagonal lattice warm-start
    points = _hexagonal_lattice(N).copy()
    
    # Projected gradient descent to maximize minimum distance
    # We'll use a soft-maximum approach to make the objective differentiable
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Iterative optimization with adaptive step size
    for iteration in range(2000):
        # Compute all pairwise distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        # Find the closest pair
        min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_dist_idx
        
        # Also find points close to boundary
        boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        min_boundary_idx = np.argmin(boundary_dists)
        
        # Adaptive step size
        step = 0.005
        
        # Push the closest pair apart
        direction = points[i] - points[j]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            points[i] += direction * step / 2
            points[j] -= direction * step / 2
        else:
            # If coincident, push randomly
            points[i] += np.array([step/2, 0])
            points[j] -= np.array([step/2, 0])
        
        # Push boundary-close points inward
        for dim in range(2):
            if points[min_boundary_idx, dim] > 0.49:
                points[min_boundary_idx, dim] -= step
            elif points[min_boundary_idx, dim] < -0.49:
                points[min_boundary_idx, dim] += step
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check improvement
        current_score = _compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        else:
            # Revert if no improvement
            points = best_points.copy()
    
    # Final refinement: local search on the best points
    for _ in range(500):
        points = best_points.copy()
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        # Find all pairs with distance close to minimum
        min_dist = np.min(dists)
        close_pairs = np.argwhere(dists < min_dist * 1.05)
        close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
        
        improved = False
        for i, j in close_pairs[:10]:  # Limit to top 10 pairs
            direction = points[i] - points[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                # Try small perturbations
                for step in [0.001, 0.002, 0.003]:
                    new_points = points.copy()
                    new_points[i] += direction * step
                    new_points[j] -= direction * step
                    new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
                    new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
                    
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
                        improved = True
                        break
        
        if not improved:
            break
    
    # Ensure final points are within bounds
    best_points[:, 0] = np.clip(best_points[:, 0], -0.5, 0.5)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.5, 0.5)
    
    return best_points
# EVOLVE-BLOCK-END
