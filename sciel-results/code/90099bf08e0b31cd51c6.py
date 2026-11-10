"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Find the centers of 50 equal non-overlapping circles in a unit square
    [-0.5, 0.5]^2 to maximize the minimum clearance (radius).
    
    Uses a multi-start approach with Nelder-Mead optimization initialized
    with jittered hexagonal lattice to maximize minimum pairwise distance
    directly, avoiding full pairwise matrix computation overhead.
    """
    
    def compute_score(centers):
        """Compute the minimum clearance (radius) for a given configuration."""
        centers = np.asarray(centers).reshape(N, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        # Boundary distances
        x, y = centers[:, 0], centers[:, 1]
        boundary_dists = np.min(np.stack([0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y], axis=1), axis=1)
        min_boundary = np.min(boundary_dists)
        
        # Radius is min(min_pairwise/2, min_boundary)
        r = min(min_pairwise / 2, min_boundary)
        return r
    
    def objective(params):
        """Minimize negative clearance for the optimizer."""
        centers = params.reshape(N, 2)
        r = compute_score(centers)
        return -r
    
    def init_hex_jittered(seed=42):
        """Jittered hexagonal lattice initialization."""
        # Estimate spacing for hex packing
        s = np.sqrt(2 / (N * np.sqrt(3)))
        
        nx = int(1 / s) + 3
        ny = int(1 / (s * np.sqrt(3) / 2)) + 3
        
        points = []
        for j in range(ny):
            y = (j + 0.5) * s * np.sqrt(3) / 2 - 0.5
            offset = (j % 2) * s / 2
            for i in range(nx):
                x = (i + 0.5) * s - 0.5 + offset
                points.append([x, y])
        
        points = np.array(points)
        dist_to_center = np.sum(points**2, axis=1)
        closest_indices = np.argsort(dist_to_center)[:N]
        centers = points[closest_indices].copy()
        
        np.random.seed(seed)
        centers += np.random.normal(0, s * 0.15, centers.shape)
        return centers
    
    def init_grid_jittered(seed=1):
        """Jittered grid initialization."""
        n = int(np.sqrt(N)) + 1
        spacing = 1.0 / n
        points = []
        for i in range(n):
            for j in range(n):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                points.append([x, y])
        
        points = np.array(points)
        dist_to_center = np.sum(points**2, axis=1)
        closest_indices = np.argsort(dist_to_center)[:N]
        centers = points[closest_indices].copy()
        
        np.random.seed(seed)
        centers += np.random.normal(0, 0.015, centers.shape)
        return centers
    
    def run_nelder_mead(x0, maxiter=5000):
        """Run Nelder-Mead optimizer and return centers and score."""
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            centers = res.x.reshape(N, 2)
            score = compute_score(centers)
            return centers, score
        except Exception:
            return None, -np.inf
    
    # Multiple initializations with different seeds for diversity
    inits = [
        init_hex_jittered(seed=42),
        init_hex_jittered(seed=123),
        init_grid_jittered(seed=1),
        init_grid_jittered(seed=7),
    ]
    
    best_centers = None
    best_score = -np.inf
    
    start_time = time.time()
    time_limit = 100  # Leave buffer for overhead
    
    for i, x0 in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        centers, score = run_nelder_mead(x0.flatten(), maxiter=5000)
        
        if centers is not None and score > best_score:
            best_score = score
            best_centers = centers
    
    # Fallback if all failed
    if best_centers is None:
        best_centers = init_hex_jittered(seed=42)
    
    # Clip to valid range strictly
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers

# EVOLVE-BLOCK-END
