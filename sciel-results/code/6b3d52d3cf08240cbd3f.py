"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hexagonal_lattice(n):
    """Generate a dense hexagonal lattice arrangement of n points in [-0.5, 0.5]^2."""
    best_points = None
    best_score = -1
    
    # Try various spacings for hexagonal lattice
    for spacing in np.linspace(0.01, 0.5, 80):
        points = []
        
        max_j = int(1.0 / (spacing * np.sqrt(3) / 2)) + 2
        max_i = int(1.0 / spacing) + 2
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        
        # Score: prefer configurations with more points and better spread
        # Use minimum pairwise distance as proxy for quality
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Also consider boundary distances
        boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        score = min(min_dist / 2, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    # Select best n points using greedy farthest-point sampling
    if len(best_points) > n:
        selected = [np.random.randint(len(best_points))]
        while len(selected) < n:
            # Find point farthest from all selected points
            min_dists = np.min([np.linalg.norm(best_points - best_points[s], axis=1) for s in selected], axis=0)
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
        best_points = best_points[selected]
    
    return best_points[:n]


def _compute_score(points):
    """Compute the packing score: min(min_pairwise_dist/2, min_boundary_dist)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2, min_boundary)


def _project(points):
    """Project points into the unit square [-0.5, 0.5]^2."""
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    return points


def construct():
    # Start with hexagonal lattice warm-start
    points = _hexagonal_lattice(N).copy()
    
    # Iterative optimization: push closest pairs apart
    # Use a simulated annealing-like approach with decreasing step size
    
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    n_iterations = 2000
    base_step = 0.02
    
    for it in range(n_iterations):
        # Annealing: decrease step size over time
        step = base_step * (1.0 - it / n_iterations) + 0.001
        
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pairs and push them apart
        # Get the 10 closest pairs for efficiency
        flat_idx = np.argsort(dists.flatten())[:10]
        
        for idx in flat_idx:
            i, j = np.unravel_index(idx, dists.shape)
            if i == j:
                continue
            
            direction = points[i] - points[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            
            direction = direction / norm
            half_step = step / 2
            
            # Move apart
            new_i = points[i] + direction * half_step
            new_j = points[j] - direction * half_step
            
            # Check if this improves the local constraint
            # Simple acceptance: always accept for now (will refine later)
            points[i] = new_i
            points[j] = new_j
        
        # Handle boundary constraints: push points away from edges
        for k in range(N):
            for dim in range(2):
                if points[k, dim] > 0.49:
                    points[k, dim] = 0.49
                elif points[k, dim] < -0.49:
                    points[k, dim] = -0.49
        
        # Project into square
        _project(points)
        
        # Compute current score
        score = _compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final refinement with scipy L-BFGS-B
    def objective(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        return -min(min_pair / 2, min_boundary)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    result = minimize(objective, best_points.flatten(), method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.fun < objective(best_points.flatten()):
        best_points = result.x.reshape(-1, 2)
    
    # Ensure points are within bounds
    _project(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
