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
    
    Uses a multi-start Powell optimization with diverse initializations
    (hexagonal lattice, grid, and Halton sequence) to escape local minima.
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
    
    def init_hex():
        """Hexagonal lattice initialization."""
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
        centers = points[closest_indices]
        
        np.random.seed(42)
        centers += np.random.normal(0, s * 0.1, centers.shape)
        return centers
    
    def init_grid():
        """Regular grid initialization."""
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
        centers = points[closest_indices]
        
        np.random.seed(1)
        centers += np.random.normal(0, 0.01, centers.shape)
        return centers
    
    def init_halton():
        """Halton sequence initialization for uniform distribution."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        points = []
        for i in range(1, N + 1):
            x = halton(i, 2) - 0.5
            y = halton(i, 3) - 0.5
            points.append([x, y])
        
        centers = np.array(points)
        np.random.seed(3)
        centers += np.random.normal(0, 0.005, centers.shape)
        return centers
    
    def run_optimizer(x0, method='Powell', maxiter=3000):
        """Run the optimizer and return centers and score."""
        try:
            res = minimize(objective, x0, method=method, 
                          options={'maxiter': maxiter, 'ftol': 1e-9, 'xtol': 1e-9})
            centers = res.x.reshape(N, 2)
            score = compute_score(centers)
            return centers, score
        except Exception:
            return None, -np.inf
    
    # Multiple initializations
    inits = [init_hex(), init_grid(), init_halton()]
    
    best_centers = None
    best_score = -np.inf
    
    start_time = time.time()
    time_limit = 100  # Leave buffer for overhead
    
    for i, x0 in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        centers, score = run_optimizer(x0.flatten(), method='Powell', maxiter=3000)
        
        if centers is not None and score > best_score:
            best_score = score
            best_centers = centers
    
    # Fallback if all failed
    if best_centers is None:
        best_centers = init_hex()
    
    # Clip to valid range strictly
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers

# EVOLVE-BLOCK-END
