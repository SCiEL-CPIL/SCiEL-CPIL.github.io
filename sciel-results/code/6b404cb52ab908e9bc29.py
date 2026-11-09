# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 16
    d = 2
    
    def compute_score(points):
        """Compute R = (d_min/d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def compute_dmin_dmax(points):
        """Compute d_min and d_max"""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        return d_min, d_max
    
    def gradient_ascent(points, lr=0.01, n_iter=5000):
        """Simple gradient ascent to maximize (d_min/d_max)^2"""
        points = points.copy()
        best_score = compute_score(points)
        best_points = points.copy()
        
        for i in range(n_iter):
            if time.time() > deadline:
                break
            
            # Compute pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max == 0:
                break
            
            # Find which pairs achieve d_min and d_max
            # For gradient, we need to know which pairs contribute to d_min and d_max
            # d_min is achieved by at least one pair, d_max by at least one pair
            
            # Compute gradient of (d_min/d_max)^2
            # Let r = d_min/d_max, we want to maximize r^2
            # dr/dr = 2r * (1/d_max * dd_min - d_min/d_max^2 * dd_max)
            # This is complex, so let's use a simpler approach:
            # Push points apart for d_min pairs, pull points together for d_max pairs
            
            # Find pairs with distance close to d_min (within 10%)
            min_pairs_mask = dists < d_min * 1.1
            # Find pairs with distance close to d_max (within 10%)
            max_pairs_mask = dists > d_max * 0.9
            
            grad = np.zeros_like(points)
            
            # For d_min pairs: push apart
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    if min_pairs_mask[i_idx, j_idx]:
                        diff_ij = points[i_idx] - points[j_idx]
                        norm = np.linalg.norm(diff_ij)
                        if norm > 1e-10:
                            grad[i_idx] += diff_ij / norm
                            grad[j_idx] -= diff_ij / norm
            
            # For d_max pairs: pull together
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    if max_pairs_mask[i_idx, j_idx]:
                        diff_ij = points[i_idx] - points[j_idx]
                        norm = np.linalg.norm(diff_ij)
                        if norm > 1e-10:
                            grad[i_idx] -= diff_ij / norm
                            grad[j_idx] += diff_ij / norm
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            points = points + lr * grad
            
            # Compute new score
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
            
            # Anneal learning rate
            lr = max(0.0001, 0.01 * (0.999 ** i))
        
        return best_points, best_score
    
    def simulated_annealing(points, n_iter=20000):
        """Simulated annealing to maximize score"""
        points = points.copy()
        best_score = compute_score(points)
        best_points = points.copy()
        
        temp = 0.1
        for i in range(n_iter):
            if time.time() > deadline:
                break
            
            # Random perturbation
            perturbed = points + np.random.randn(n, d) * temp
            
            score = compute_score(perturbed)
            current_score = compute_score(points)
            
            if score > current_score:
                points = perturbed
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
            else:
                # Accept with probability exp((score - current_score) / temp)
                if np.random.random() < np.exp((score - current_score) / (temp + 1e-10)):
                    points = perturbed
            
            temp = max(0.001, 0.1 * (0.9995 ** i))
        
        return best_points, best_score
    
    # Try multiple initial configurations
    best_points = None
    best_score = -1
    
    # Configuration 1: Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Configuration 2: 4x4 grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    config2 = np.column_stack([gx.flatten(), gy.flatten()])
    
    # Configuration 3: Two concentric squares (8 points each)
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)]) * 1.5
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)]) * 0.7
    config3 = np.vstack([outer, inner])
    
    # Configuration 4: 4 points on circle, 12 in 3x4 grid pattern
    # Let's try: 4 corners of square + 4 midpoints + 4 more + center region
    # Actually, let's try a different combination: 2 concentric 8-point sets with different radii
    angles8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer8 = np.column_stack([np.cos(angles8), np.sin(angles8)]) * 1.3
    inner8 = np.column_stack([np.cos(angles8 + np.pi/8), np.sin(angles8 + np.pi/8)]) * 0.6
    config4 = np.vstack([outer8, inner8])
    
    configs = [config1, config2, config3, config4]
    
    for idx, config in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        score = compute_score(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
        
        # Run gradient ascent
        if time.time() > deadline:
            break
        optimized, opt_score = gradient_ascent(config, lr=0.05, n_iter=3000)
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()
        
        # Run simulated annealing from the optimized point
        if time.time() > deadline:
            break
        sa_points, sa_score = simulated_annealing(optimized, n_iter=10000)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # If we have time, do a final refinement
    if best_points is not None and time.time() < deadline - 30:
        final_points, final_score = gradient_ascent(best_points, lr=0.01, n_iter=5000)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()
    
    if best_points is None:
        # Fallback: regular 16-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
