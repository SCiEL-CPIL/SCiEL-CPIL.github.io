"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math

def _covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi

def _generate_grid(num_points=20000):
    """Generate a fixed grid of points on the sphere for gradient estimation."""
    # Use Fibonacci spiral for the grid too, for uniformity
    i = np.arange(num_points) + 0.5
    z = 1.0 - 2.0 * i / num_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    grid = _generate_grid(20000)
    
    # Subgradient Descent parameters
    num_iterations = 2000
    learning_rate = 0.05
    weight_decay = 0.99
    beta = 2.0 # Temperature for weighting worst points
    
    best_points = points.copy()
    best_energy = _covering_radius(points)
    
    for t in range(num_iterations):
        # Compute squared distances: (M, N)
        # dist^2 = |G|^2 + |P|^2 - 2 G.P = 1 + 1 - 2 G.P = 2 - 2 G.P
        # Since G and P are unit vectors.
        G_P = grid @ points.T # (M, N)
        dist_sq = 2.0 - 2.0 * G_P
        
        # Find nearest point index for each grid point
        nearest_idx = np.argmin(dist_sq, axis=1)
        
        # Compute weights based on distance (focus on large holes)
        min_dist_sq = dist_sq[np.arange(len(grid)), nearest_idx]
        
        # Weights: exp(beta * dist_sq)
        # Normalize weights to sum to 1 to keep step size stable
        weights = np.exp(beta * min_dist_sq)
        weights /= np.sum(weights)
        
        # Compute gradient
        # grad_P_j = sum_{i: nearest(i)=j} 2 * weights[i] * (P_j - G_i)
        # We can accumulate this efficiently
        grad = np.zeros_like(points)
        
        # For each grid point, add contribution to its nearest point's gradient
        # grad[nearest_idx] += 2 * weights[:, None] * (points[nearest_idx] - grid)
        
        # This requires advanced indexing which might be slow in a loop, 
        # but vectorized addition is tricky with repeated indices.
        # Use np.add.at
        contributions = 2.0 * weights[:, None] * (points[nearest_idx] - grid)
        np.add.at(grad, nearest_idx, contributions)
        
        # Update points
        # Adjust learning rate
        lr = learning_rate * (0.5 ** (t / 500.0))
        
        points = points - lr * grad
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Evaluate occasionally
        if t % 50 == 0:
            current_energy = _covering_radius(points)
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
                
    # Final polish
    final_energy = _covering_radius(best_points)
    
    # Ensure valid output
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
