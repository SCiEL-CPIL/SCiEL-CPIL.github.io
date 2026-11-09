# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    # Initialize points on a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Optimization parameters
    n_iterations = 10000
    initial_lr = 0.05
    final_lr = 0.0001
    
    for it in range(n_iterations):
        # Anneal learning rate
        lr = initial_lr * (final_lr / initial_lr) ** (it / n_iterations)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        
        # For each point, find the nearest neighbor
        np.fill_diagonal(dists, np.inf)
        nearest_indices = np.argmin(dists, axis=1)
        
        # Compute update vectors
        updates = np.zeros_like(points)
        for i in range(n):
            j = nearest_indices[i]
            vec = points[i] - points[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                updates[i] += vec / norm
        
        # Update points
        points += lr * updates
        
        # Center
        points -= points.mean(axis=0)
        
        # Scale to unit max distance
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        d_max = np.max(dists)
        if d_max > 1e-10:
            points /= d_max
    
    return points

# EVOLVE-BLOCK-END
