"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    """
    Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum distance.
    
    Strategy:
    1. Initialize with a hexagonal lattice pattern (better than square for packing).
    2. Use a repulsion-based optimization with multiple restarts.
    3. Use simulated annealing / gradient ascent on the minimum distance.
    4. Return the best configuration found within time budget.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    def score(centers):
        """Compute the minimum distance (which equals 2*r)."""
        if len(centers) < 2:
            return np.inf
        tree = cKDTree(centers)
        pairs, dists = tree.query_pairs(1.0, output_type='ndist')
        if len(pairs) > 0:
            min_pair_dist = min(dists)
        else:
            min_pair_dist = np.inf
        # Distance to boundary
        min_boundary_dist = np.min(np.minimum(np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
                                               np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)))
        return min(min_pair_dist, 2 * min_boundary_dist)
    
    def hexagonal_init(n_points):
        """Initialize points in a hexagonal lattice pattern."""
        # For 200 points, use roughly a 14x15 grid with hexagonal offset
        rows = 14
        cols = 15
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = (j + 0.5 * (i % 2)) / cols - 0.5
                y = i / rows - 0.5
                points.append([x, y])
        return np.array(points[:n_points])
    
    def random_init(n_points):
        """Random initialization inside the square."""
        return np.random.uniform(-0.5, 0.5, size=(n_points, 2))
    
    def optimize(centers, max_iters=5000, lr=0.01, deadline=None):
        """
        Optimize centers using gradient ascent on the minimum distance.
        Uses a repulsion model to push points apart.
        """
        n = len(centers)
        centers = centers.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            # Compute distances to all other points and boundaries
            tree = cKDTree(centers)
            
            # Find closest pair
            pairs, dists = tree.query_pairs(1.0, output_type='ndist')
            
            if len(pairs) == 0:
                min_dist = np.inf
            else:
                min_dist = min(dists)
            
            # Boundary distances
            boundary_dists = np.minimum(np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
                                        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5))
            min_boundary = np.min(boundary_dists)
            
            current_score = min(min_dist, 2 * min_boundary)
            
            # Compute gradient: push points apart
            grad = np.zeros_like(centers)
            
            # Pairwise repulsion
            for i, j in pairs:
                if dists[i, j] < current_score + 0.01:  # Only consider close pairs
                    diff = centers[i] - centers[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        diff /= norm
                        grad[i] += diff
                        grad[j] -= diff
            
            # Boundary repulsion
            for k in range(n):
                # Left boundary
                if centers[k, 0] + 0.5 < current_score / 2 + 0.01:
                    grad[k, 0] += 1.0
                # Right boundary
                if 0.5 - centers[k, 0] < current_score / 2 + 0.01:
                    grad[k, 0] -= 1.0
                # Bottom boundary
                if centers[k, 1] + 0.5 < current_score / 2 + 0.01:
                    grad[k, 1] += 1.0
                # Top boundary
                if 0.5 - centers[k, 1] < current_score / 2 + 0.01:
                    grad[k, 1] -= 1.0
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad /= grad_norm
            
            # Update with adaptive learning rate
            step_size = lr * (1.0 + 0.1 * np.sin(it * 0.01))
            new_centers = centers + step_size * grad
            
            # Ensure points stay within bounds
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check if improvement
            new_score = score(new_centers)
            if new_score > current_score:
                centers = new_centers
            else:
                # Add some randomness to escape local minima
                if it % 100 == 0:
                    noise = np.random.normal(0, 0.001, size=centers.shape)
                    centers = centers + noise
                    centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                    centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        return centers, score(centers)
    
    best_centers = None
    best_score = 0
    
    # Multiple restarts with different initializations
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            centers = hexagonal_init(N)
        elif restart == 1:
            centers = np.random.uniform(-0.45, 0.45, size=(N, 2))
        else:
            # Perturbed hexagonal
            centers = hexagonal_init(N)
            centers += np.random.normal(0, 0.02, size=centers.shape)
            centers[:, 0] = np.clip(centers[:, 0], -0.48, 0.48)
            centers[:, 1] = np.clip(centers[:, 1], -0.48, 0.48)
        
        centers, sc = optimize(centers, max_iters=3000, lr=0.005, deadline=deadline)
        
        if sc > best_score:
            best_score = sc
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = hexagonal_init(N)
    
    return best_centers
# EVOLVE-BLOCK-END
