# EVOLVE-BLOCK-START
import numpy as np
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    def fibonacci_sphere(n_points):
        """Generate points on a sphere using Fibonacci lattice."""
        phi = np.arccos(1 - 2 * (np.arange(n_points) + 0.5) / n_points)
        theta = np.pi * (1 + 5**0.5) * np.arange(n_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))

    def optimize_points(points, iterations=500, lr=0.01):
        """
        Optimize points on the unit sphere to maximize min distance
        by minimizing the sum of inverse squared distances (repulsion energy).
        """
        pts = points.copy()
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        for i in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Calculate distances
            # pts shape (n, 3)
            # diff shape (n, n, 3)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Avoid division by zero
            dist_sq[dist_sq < 1e-10] = 1e-10
            
            # Energy: sum 1/dist_sq for i<j
            # Gradient of 1/dist_sq w.r.t pts[i]
            # d/dx (1/||x-y||^2) = -2(x-y)/||x-y||^4
            # We want to minimize E, so we move in direction -grad E
            # grad E_i = sum_{j != i} -2 (pts[i] - pts[j]) / dist_sq[i,j]^2
            
            # Force matrix F[i,j] = -2 * (pts[i] - pts[j]) / dist_sq[i,j]^2
            # Note: dist_sq[i,j] is ||pts[i]-pts[j]||^2
            # So dist^4 is dist_sq^2
            
            # Create mask for i != j
            mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mask, False)
            
            # Calculate forces
            # term = diff / dist_sq^2
            # diff is (n,n,3), dist_sq is (n,n)
            # We need to divide each vector by the scalar dist_sq^2
            forces = np.zeros_like(pts)
            
            # Efficient calculation
            # dist_sq_inv_sq = 1 / (dist_sq ** 2)
            dist_sq_inv_sq = 1.0 / (dist_sq ** 2)
            np.fill_diagonal(dist_sq_inv_sq, 0.0)
            
            # Force on i from j: -2 * (p_i - p_j) * (1/dist_sq_ij^2)
            # Sum over j
            # forces[i] = -2 * sum_j (p_i - p_j) * dist_sq_inv_sq[i,j]
            
            # We can compute this as:
            # sum_j p_j * dist_sq_inv_sq[i,j] - p_i * sum_j dist_sq_inv_sq[i,j]
            
            # Let W be dist_sq_inv_sq
            # sum_j p_j W[i,j] is pts.T @ W? No, pts is (n,3), W is (n,n).
            # We want vector v_i = sum_j W[i,j] * p_j
            # This is (W @ pts).T ? 
            # (W @ pts) has shape (n, 3) if W is (n,n) and pts is (n,3).
            # Row i of (W @ pts) is sum_j W[i,j] * pts[j].
            # So v = W @ pts.
            
            v = dist_sq_inv_sq @ pts
            row_sums = np.sum(dist_sq_inv_sq, axis=1, keepdims=True)
            
            # Force on i: -2 * (v_i - p_i * row_sum_i)
            # Wait, the formula was:
            # grad E_i = sum_{j != i} -2 (p_i - p_j) / d_{ij}^4
            # = -2 * [ p_i * sum_{j != i} 1/d_{ij}^4 - sum_{j != i} p_j / d_{ij}^4 ]
            # = -2 * [ p_i * row_sum_i - v_i ]
            # = 2 * [ v_i - p_i * row_sum_i ]
            
            forces = 2.0 * (v - pts * row_sums)
            
            # Update
            pts = pts + lr * forces
            
            # Project back to unit sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            # Avoid zero norm
            norms[norms < 1e-10] = 1e-10
            pts = pts / norms
            
        return pts

    def calculate_score(points):
        """Calculate (d_min / d_max)^2"""
        from scipy.spatial.distance import pdist
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -1.0
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Fibonacci Sphere
    initial_configs.append(fibonacci_sphere(n))
    
    # 2. Random configurations
    rng = np.random.default_rng(42)
    for i in range(10):
        # Random points on sphere
        pts = rng.normal(size=(n, d))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)
        
        # Check time
        if time.time() > deadline - 10:
            break

    # 3. Perturbed Icosahedron (12 vertices) + 2 random? 
    # Let's just stick to the above. 11 configs.
    
    # Optimize each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break
            
        # Optimize
        # Use a slightly larger learning rate for initial phase? 
        # Let's try a few iterations with different LRs or just one robust run.
        opt_pts = optimize_points(init_pts, iterations=300, lr=0.05)
        
        # Polish with smaller LR
        opt_pts = optimize_points(opt_pts, iterations=300, lr=0.005)
        
        # Evaluate
        score = calculate_score(opt_pts)
        
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
            
    # If no points were found (should not happen), return random
    if best_points is None:
        best_points = fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
