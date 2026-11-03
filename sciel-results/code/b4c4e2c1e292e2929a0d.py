import numpy as np

N = 100

def construct():
    def fibonacci_points(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pts = np.column_stack((x, y, z))
        return pts

    def energy_and_grad(pts):
        # pts shape (N, 3)
        # Compute pairwise squared distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        sq_dists[np.diag_indices_from(sq_dists)] = 1.0
        
        dists = np.sqrt(sq_dists)
        
        # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # Using symmetry: sum_{i<j} = 0.5 * sum_{i!=j}
        # log(1/d) = -log(d)
        # So E = -0.5 * sum_{i!=j} log(d_ij)
        
        # Gradient:
        # dE/dx_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
        # Let's compute the sum term S_i = sum_{j != i} (x_i - x_j) / sq_dists[i,j]
        
        # We can compute this using matrix operations
        # diff is (N, N, 3)
        # inv_sq_dists is (N, N)
        inv_sq_dists = 1.0 / sq_dists
        # Set diagonal to 0
        np.fill_diagonal(inv_sq_dists, 0.0)
        
        # grad_unproj[i] = sum_j diff[i,j] * inv_sq_dists[i,j]
        # diff[i,j] = x_i - x_j
        # So grad_unproj[i] = sum_j (x_i - x_j) / ||x_i - x_j||^2
        grad_unproj = np.sum(diff * inv_sq_dists[:, :, np.newaxis], axis=1)
        
        # Energy calculation
        # Mask for i < j
        iu = np.triu_indices(N, k=1)
        d_upper = dists[iu]
        # Avoid log(0) if points are very close
        d_upper = np.maximum(d_upper, 1e-12)
        E = -np.sum(np.log(d_upper))
        
        return E, grad_unproj

    def optimize(pts):
        # pts is (N, 3)
        # Gradient descent with projection
        lr = 1e-4
        # Simple decay
        # Run for a fixed number of iterations or until convergence
        # 120s limit. N=100. 
        # Each iteration is O(N^2). 100^2 = 10k ops. 
        # 10k * 1000 iterations = 10M ops. Fast.
        
        for it in range(2000):
            E, grad = energy_and_grad(pts)
            
            # Project gradient to tangent space
            # grad_proj = grad - (grad . pts) * pts
            dot = np.sum(grad * pts, axis=1, keepdims=True)
            grad_proj = grad - dot * pts
            
            # Update
            pts = pts + lr * grad_proj
            
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            # Decay learning rate
            if it % 200 == 199:
                lr *= 0.5
                
        return pts, E

    best_pts = None
    best_E = np.inf
    
    # Multiple restarts
    # 1. Fibonacci
    # 2-5. Fibonacci + Noise
    
    for k in range(5):
        pts = fibonacci_points(N)
        if k > 0:
            # Add random noise
            noise = np.random.normal(0, 0.1 * k, size=pts.shape)
            pts = pts + noise
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
        
        # Optimize
        # Use a fixed random seed for reproducibility in evaluation?
        # The task doesn't specify, but usually deterministic is preferred or we just use default.
        # Let's use a fixed seed for the noise to ensure consistent runs if needed, 
        # or just let it vary. The evaluator runs it once.
        np.random.seed(42 + k)
        
        res_pts, res_E = optimize(pts)
        
        if res_E < best_E:
            best_E = res_E
            best_pts = res_pts
            
    return best_pts
