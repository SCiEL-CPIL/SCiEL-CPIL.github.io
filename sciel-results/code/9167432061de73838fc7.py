# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a soft-min approximation for the minimum distance and a log-barrier
    for the maximum distance constraint, optimized with BFGS.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    def compute_score(points):
        """Compute the metric: (d_min / d_max)^2"""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        return (d_min / d_max) ** 2
    
    def objective_and_grad(x, mu=1.0, beta=0.01):
        """
        x is flattened array of shape (n*d,)
        Objective: -softmin(pair_sq_dists) + mu * sum(-log(1 - pair_sq_dists))
        """
        points = x.reshape(n, d)
        
        # Compute all pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        
        # Extract pairwise squared distances (upper triangle)
        pair_sq_dists = sq_dists[i_idx, j_idx]
        
        # Check if any distance is too close to 1
        if np.any(pair_sq_dists >= 0.999999):
            return 1e10, np.zeros_like(x)
        
        # Soft-min of pair_sq_dists
        # softmin(y) = -beta * log(sum(exp(-y_i / beta)))
        # We want to MAXIMIZE min, so we MINIMIZE -min.
        # So objective part 1 is -softmin(y) = beta * log(sum(exp(-y_i / beta)))
        
        # For numerical stability, subtract max of -y_i/beta (which is -min(y_i)/beta)
        # Let z = -pair_sq_dists / beta
        z = -pair_sq_dists / beta
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        sum_exp_z = np.sum(exp_z)
        
        soft_min_val = -beta * (z_max + np.log(sum_exp_z))
        # We minimize -soft_min_val
        obj_part1 = -soft_min_val
        
        # Barrier term
        barrier = -np.sum(np.log(1.0 - pair_sq_dists))
        obj_part2 = mu * barrier
        
        obj = obj_part1 + obj_part2
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Gradient of obj_part1:
        # d/dx [-softmin(y)] = d/dx [beta * log(sum(exp(z)))]
        # = beta * (1/sum(exp(z))) * sum(exp(z) * dz/dx)
        # dz/dx = -1/beta * dy/dx
        # So grad = (1/sum(exp(z))) * sum(exp(z) * (-dy/dx))
        # = - sum( w_i * dy_i/dx ) where w_i = exp(z_i) / sum(exp(z))
        
        weights = exp_z / sum_exp_z # shape (n_pairs,)
        
        # dy_i/dx for pair (i,j) is:
        # y_i = ||p_i - p_j||^2
        # d y_i / d p_i = 2(p_i - p_j)
        # d y_i / d p_j = -2(p_i - p_j)
        
        # We need sum_i w_i * (2(p_i - p_j)) for p_i and -w_i * (2(p_i - p_j)) for p_j
        
        pair_diffs = points[i_idx] - points[j_idx] # shape (n_pairs, d)
        
        # Weighted diffs for i_idx
        weighted_diffs_i = 2.0 * weights[:, np.newaxis] * pair_diffs
        # Accumulate to points at i_idx
        grad_reshaped = grad.reshape(n, d)
        np.add.at(grad_reshaped, i_idx, weighted_diffs_i)
        
        # Weighted diffs for j_idx
        # Contribution is -2 * w_i * (p_i - p_j)
        np.add.at(grad_reshaped, j_idx, -weighted_diffs_i)
        
        # Gradient of obj_part2 (Barrier):
        # d/dx [mu * sum(-log(1 - y_i))] = mu * sum( 1/(1-y_i) * dy_i/dx )
        # = mu * sum( 1/(1-y_i) * 2(p_i - p_j) for p_i ... )
        
        barrier_weights = mu * 2.0 / (1.0 - pair_sq_dists) # shape (n_pairs,)
        weighted_diffs_barrier = barrier_weights[:, np.newaxis] * pair_diffs
        
        np.add.at(grad_reshaped, i_idx, weighted_diffs_barrier)
        np.add.at(grad_reshaped, j_idx, -weighted_diffs_barrier)
        
        return obj, grad.flatten()

    # Generate multiple initial guesses
    best_points = None
    best_score = -1
    
    np.random.seed(42)
    
    # Multiple random starts
    for trial in range(50):
        x0 = np.random.randn(n * d) * 0.3
        
        # Also try structured starts
        if trial == 0:
            # Fibonacci sphere
            golden_ratio = (1 + np.sqrt(5)) / 2
            y = np.linspace(-1, 1, n)
            r = np.sqrt(np.maximum(0, 1 - y**2))
            theta = 2 * np.pi * np.arange(n) / golden_ratio
            sphere_pts = np.column_stack([np.cos(theta) * r, np.sin(theta) * r, y])
            x0 = sphere_pts.flatten()
        elif trial == 1:
            # Icosahedron + 2 extra
            phi = (1 + np.sqrt(5)) / 2
            ico_verts = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ])
            ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
            extra = np.array([[0, 0, 0], [0.3, 0, 0]])
            ico_plus = np.vstack([ico_verts, extra])
            x0 = ico_plus.flatten()
        elif trial == 2:
            # Cube vertices + face centers
            cube_verts = np.array([
                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
                [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                [0, 0, -1], [0, 0, 1]
            ])
            cube_verts = cube_verts / np.linalg.norm(cube_verts, axis=1, keepdims=True)
            x0 = cube_verts.flatten()
            
        try:
            # Use a smaller beta for sharper min, but larger for smoothness.
            # Start with moderate beta.
            result = minimize(
                objective_and_grad,
                x0,
                method='BFGS',
                jac=True,
                args=(1.0, 0.05), # mu, beta
                options={'maxiter': 3000, 'gtol': 1e-12, 'disp': False}
            )
            
            points = result.x.reshape(n, d)
            
            # Check validity
            if not np.all(np.isfinite(points)):
                continue
            
            score = compute_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue
    
    # If no valid result, fallback
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    # Center the points
    centroid = np.mean(best_points, axis=0)
    best_points = best_points - centroid
    
    return best_points


# EVOLVE-BLOCK-END
