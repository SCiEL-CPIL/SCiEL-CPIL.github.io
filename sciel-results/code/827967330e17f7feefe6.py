import numpy as np
from scipy.optimize import minimize_scalar

def construct():
    # Define the two-square configuration
    # Square 1 (outer, radius 1)
    # Square 2 (inner, radius r)
    
    def get_points(r):
        # Outer square
        p1 = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
        # Inner square, rotated by 45 degrees
        angle = np.pi / 4
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        base_inner = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
        p2 = (base_inner @ rot.T) * r
        return np.vstack((p1, p2))

    def neg_log_product(r):
        pts = get_points(r)
        # Calculate all pairwise distances
        n = 8
        # Use broadcasting to compute distance matrix
        # dist[i, j] = ||pts[i] - pts[j]||
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal and upper triangle to avoid double counting and zeros
        # We want product of all distinct pairs
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_dists = dists[mask]
        # Avoid log(0) just in case, though points are distinct
        valid_dists = np.maximum(valid_dists, 1e-10)
        log_prod = np.sum(np.log(valid_dists))
        return -log_prod

    # Optimize r in (0, 1]
    # Use bounded scalar minimization
    res = minimize_scalar(neg_log_product, bounds=(0.1, 1.0), method='bounded', options={'xatol': 1e-10})
    
    r_opt = res.x
    optimal_pts = get_points(r_opt)
    
    # Convert to list of lists of floats
    return optimal_pts.tolist()
