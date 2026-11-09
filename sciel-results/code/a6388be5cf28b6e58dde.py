# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def calculate_metric(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def angles_to_cartesian(angles):
        # angles shape (n, 2): theta, phi
        theta = angles[:, 0]
        phi = angles[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))

    def objective(angles):
        pts = angles_to_cartesian(angles)
        dists = pdist(pts)
        if np.any(dists < 1e-10):
            return 1e10
        # We want to maximize min distance. 
        # Equivalent to minimizing -min distance.
        # -min is non-smooth. 
        # Use sum of inverse squares as proxy, which is smooth and pushes all distances up.
        return np.sum(1.0 / (dists ** 2))

    # Bounds for theta: [0, pi], phi: [0, 2*pi]
    bounds = [(0, np.pi)] * n + [(0, 2 * np.pi)] * n
    
    # Differential Evolution
    # maxiter and popsize control the search
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=200, 
        popsize=15, 
        seed=42,
        tol=1e-4,
        disp=False
    )
    
    best_angles = result.x.reshape(n, 2)
    best_pts = angles_to_cartesian(best_angles)
    
    return best_pts

# EVOLVE-BLOCK-END
