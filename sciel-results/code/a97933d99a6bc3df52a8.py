import numpy as np
import time
import sys

N = 130

def construct():
    """
    Covers the unit sphere with 130 equal caps by minimizing the covering radius.
    
    Strategy: Treat the 130 points as repulsive charges on a fixed high-resolution
    spherical mesh. Use a differentiable soft-minimum of the distance field to compute
    a smooth gradient that pushes points specifically toward locations of maximum
    grid coverage error. This is a continuous density minimization approach.
    
    The algorithm:
    1. Initialize with a Fibonacci spiral
    2. Iteratively:
       a. Compute the soft-minimum distance from each mesh point to the nearest configuration point
       b. Compute the gradient of the soft-max of these distances w.r.t. each configuration point
       c. Move each point in the direction that reduces the soft-max covering radius
       d. Normalize points to stay on the sphere
    3. Use adaptive step sizes and multiple restarts to escape local minima
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for the 120s limit
    
    def fibonacci_spiral(n):
        """Generate n points on the unit sphere using a Fibonacci spiral."""
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    def generate_mesh(n_lat, n_lon):
        """Generate a spherical mesh of test points."""
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        z = np.sin(LA)
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    def soft_min_dist(dists, beta):
        """
        Compute soft-minimum of distances and its gradient w.r.t. dists.
        
        soft_min(d) = -1/beta * log(sum(exp(-beta * d_i)))
        d(soft_min)/d(d_j) = -exp(-beta * d_j) / sum(exp(-beta * d_i))
        
        Parameters:
        -----------
        dists : (M, N) array of distances from M mesh points to N config points
        beta : temperature parameter (higher = closer to true min)
        
        Returns:
        --------
        soft_min_vals : (M,) array of soft-min distances
        weights : (M, N) array of weights for gradient computation
        """
        # Subtract max for numerical stability
        max_d = np.max(dists, axis=1, keepdims=True)
        exp_terms = np.exp(-beta * (dists - max_d))
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        
        # soft_min = -1/beta * log(sum(exp(-beta * d))) + max_d
        # But we want the actual soft-min value:
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability:
        # soft_min = max_d - 1/beta * log(sum(exp(-beta * (d - max_d))))
        soft_min_vals = max_d.squeeze() - (1.0/beta) * np.log(sum_exp.squeeze())
        
        # Weights: w_j = exp(-beta * d_j) / sum(exp(-beta * d_i))
        # This gives the gradient contribution
        weights = exp_terms / sum_exp
        
        return soft_min_vals, weights
    
    def compute_gradient(points, mesh, beta):
        """
        Compute the gradient of the soft-max covering radius w.r.t. each point.
        
        The objective is: F = soft_max_i(soft_min_j(d_ij))
        where d_ij is the distance from mesh point i to config point j.
        
        For each config point j, the gradient is:
        grad_j = sum_i (w_i * v_ij) / ||v_ij||^2 * (1 - d_ij / ||v_ij||)
        
        where v_ij = mesh_i - points_j, and w_i are the soft-max weights.
        
        Parameters:
        -----------
        points : (N, 3) array of configuration points
        mesh : (M, 3) array of mesh points
        beta : temperature parameter
        
        Returns:
        --------
        grad : (N, 3) gradient array
        """
        M = len(mesh)
        
        # Compute distances: (M, N)
        dots = np.dot(mesh, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Soft-min distances and weights
        soft_min_vals, min_weights = soft_min_dist(dists, beta)
        
        # Soft-max weights for the outer max
        max_sm = np.max(soft_min_vals)
        exp_sm = np.exp(beta * (soft_min_vals - max_sm))
        sum_exp_sm = np.sum(exp_sm)
        max_weights = exp_sm / sum_exp_sm  # (M,)
        
        # Compute gradient for each point
        grad = np.zeros_like(points)
        
        # For each mesh point, contribute to the gradient of each config point
        # Weighted by max_weights[i] and min_weights[i, j]
        
        # v_ij = mesh_i - points_j, ||v_ij|| = 2*sin(d_ij/2)
        # d(soft_min)/d(points_j) involves the gradient of d_ij w.r.t. points_j
        # d(d_ij)/d(points_j) = -v_ij / ||v_ij|| (for small angles, approximately)
        
        # More precisely: d_ij = arccos(mesh_i · points_j)
        # d(d_ij)/d(points_j) = -mesh_i / sqrt(1 - (mesh_i · points_j)^2)
        # But we need to be careful about the sign and normalization.
        
        # Actually, for a point p on the unit sphere, the gradient of the angular
        # distance to a fixed point m is:
        # grad_p d(m,p) = -(m - (m·p)p) / ||m - (m·p)p||
        # This is the unit vector from p away from m, projected onto the tangent plane.
        
        # Let's compute this properly:
        # For each pair (i, j), the gradient of d_ij w.r.t. points_j is:
        # g_ij = -(mesh_i - (mesh_i · points_j) * points_j) / ||mesh_i - (mesh_i · points_j) * points_j||
        
        # We'll compute this efficiently:
        # mesh_i - (mesh_i · points_j) * points_j = mesh_i - dots[i,j] * points_j
        
        # To avoid memory issues, process in chunks
        chunk_size = 500
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            mesh_chunk = mesh[start:end]  # (chunk, 3)
            dots_chunk = dots[start:end]  # (chunk, N)
            min_w_chunk = min_weights[start:end]  # (chunk, N)
            max_w_chunk = max_weights[start:end]  # (chunk,)
            
            # Compute projection: mesh_i - dots[i,j] * points_j
            # Shape: (chunk, N, 3)
            proj = mesh_chunk[:, np.newaxis, :] - dots_chunk[:, :, np.newaxis] * points[np.newaxis, :, :]
            
            # Norm of projection
            proj_norm = np.linalg.norm(proj, axis=2)  # (chunk, N)
            proj_norm = np.maximum(proj_norm, 1e-10)
            
            # Gradient direction: -proj / proj_norm
            grad_dir = -proj / proj_norm[:, :, np.newaxis]  # (chunk, N, 3)
            
            # Weight: max_w_chunk[i] * min_w_chunk[i,j]
            # Shape: (chunk, N)
            weight = max_w_chunk[:, np.newaxis] * min_w_chunk
            
            # Sum contributions: sum_i weight[i,j] * grad_dir[i,j]
            # grad[j] += sum_i weight[i,j] * grad_dir[i,j]
            grad_contrib = np.sum(weight[:, :, np.newaxis] * grad_dir, axis=0)  # (N, 3)
            grad += grad_contrib
        
        return grad, soft_min_vals
    
    def normalize(points):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return points / norms
    
    def true_covering_radius(points, test_mesh):
        """Compute the true covering radius (max of min distances)."""
        dots = np.dot(test_mesh, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    # Generate high-resolution mesh for accurate gradient computation
    n_lat = 60
    n_lon = 120
    mesh = generate_mesh(n_lat, n_lon)
    
    # Also generate a coarser mesh for fast evaluation
    n_lat_eval = 40
    n_lon_eval = 80
    eval_mesh = generate_mesh(n_lat_eval, n_lon_eval)
    
    best_points = None
    best_radius = float('inf')
    
    # Multiple restarts to escape local minima
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize
        if restart == 0:
            points = fibonacci_spiral(N)
        else:
            # Perturb the best solution found so far
            if best_points is not None:
                points = best_points.copy()
                # Add random perturbation
                noise_scale = 0.1 * (restart + 1)
                noise = np.random.randn(N, 3) * noise_scale
                points = points + noise
                points = normalize(points)
            else:
                points = fibonacci_spiral(N)
        
        # Adaptive optimization
        beta = 20.0  # Start with moderate temperature
        lr = 0.05  # Learning rate
        patience = 0
        max_patience = 50
        
        for iteration in range(300):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient
            grad, soft_min_vals = compute_gradient(points, mesh, beta)
            
            # Normalize gradient to have unit norm per point for stable updates
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms = np.maximum(grad_norms, 1e-10)
            grad_unit = grad / grad_norms
            
            # Update points
            points = points + lr * grad_unit
            points = normalize(points)
            
            # Evaluate true covering radius periodically
            if iteration % 10 == 0 or iteration == 299:
                current_radius = true_covering_radius(points, eval_mesh)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                    patience = 0
                else:
                    patience += 1
                
                # Decay learning rate
                lr = max(0.001, lr * 0.98)
                
                # Increase beta (sharper min) as we converge
                beta = min(100.0, beta * 1.02)
                
                # Early stopping
                if patience >= max_patience:
                    break
        
        # Final polish with higher resolution
        if time.time() > deadline - 15.0:
            break
        
        if best_points is not None:
            # Do a few more iterations with higher beta
            points = best_points.copy()
            beta_high = 50.0
            lr_high = 0.01
            
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                grad, _ = compute_gradient(points, mesh, beta_high)
                grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norms = np.maximum(grad_norms, 1e-10)
                grad_unit = grad / grad_norms
                
                points = points + lr_high * grad_unit
                points = normalize(points)
                
                current_radius = true_covering_radius(points, eval_mesh)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                else:
                    lr_high *= 0.95
    
    if best_points is None:
        best_points = fibonacci_spiral(N)
    
    return best_points

# EVOLVE-BLOCK-END
