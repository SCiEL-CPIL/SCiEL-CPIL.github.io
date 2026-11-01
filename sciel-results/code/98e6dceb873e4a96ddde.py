import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral"""
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points
    
    def random_sphere(n, seed=42):
        """Generate random points on sphere"""
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points
    
    def compute_energy(pts):
        """Compute logarithmic energy: sum_{i<j} log(1/d_ij)"""
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
        dist_sq = np.clip(dist_sq, 1e-15, None)
        dist = np.sqrt(dist_sq)
        i_upper, j_upper = np.triu_indices(N, k=1)
        d_pairs = dist[i_upper, j_upper]
        return -np.sum(np.log(d_pairs))
    
    def compute_gradient(pts):
        """Compute gradient of energy w.r.t. points"""
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
        dist_sq = np.clip(dist_sq, 1e-15, None)
        
        diff = pts[:, None, :] - pts[None, :, :]
        inv_d2 = 1.0 / dist_sq
        np.fill_diagonal(inv_d2, 0.0)
        
        row_sums = np.sum(inv_d2, axis=1)
        grad = pts * row_sums[:, None] - inv_d2 @ pts
        return grad
    
    def project_to_sphere(pts):
        """Project points to unit sphere"""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-15, None)
        return pts / norms
    
    def optimize_lbfgs(pts, max_iter=500, time_limit=None):
        """Optimize using L-BFGS-B with sphere constraint via parameterization"""
        from scipy.optimize import minimize
        
        def objective_and_grad(x):
            # x is flattened (N*3,)
            pts = x.reshape(N, 3)
            # Project to sphere for computation
            pts = project_to_sphere(pts)
            energy = compute_energy(pts)
            grad = compute_gradient(pts)
            # Project gradient to tangent space: grad - (grad·x)x
            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            return energy, grad_tangent.flatten()
        
        x0 = pts.flatten()
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 2}
        )
        
        best_pts = project_to_sphere(result.x.reshape(N, 3))
        best_energy = compute_energy(best_pts)
        
        return best_pts, best_energy
    
    def polish_steepest_descent(pts, max_iter=200, time_limit=None):
        """Final polish with steepest descent in tangent space"""
        current_energy = compute_energy(pts)
        step_size = 1e-4
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
                
            grad = compute_gradient(pts)
            # Project to tangent space
            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-12:
                break
            
            # Normalize gradient direction
            direction = -grad_tangent / grad_norm
            
            # Try step
            new_pts = pts + step_size * direction
            new_pts = project_to_sphere(new_pts)
            new_energy = compute_energy(new_pts)
            
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                step_size = min(step_size * 1.05, 1e-2)
            else:
                step_size *= 0.7
                if step_size < 1e-10:
                    break
        
        return pts, current_energy
    
    # Try multiple initializations
    best_points = None
    best_energy = float('inf')
    
    # Initialization strategies
    inits = [
        ("fibonacci", fibonacci_sphere(N)),
        ("random1", random_sphere(N, seed=1)),
        ("random2", random_sphere(N, seed=42)),
        ("random3", random_sphere(N, seed=123)),
    ]
    
    for name, init_pts in inits:
        if time.time() > deadline - 30:
            break
        
        # L-BFGS-B optimization
        try:
            opt_pts, opt_energy = optimize_lbfgs(init_pts, max_iter=300, time_limit=deadline - 15)
        except Exception:
            opt_pts, opt_energy = init_pts, compute_energy(init_pts)
        
        # Polish
        try:
            polished_pts, polished_energy = polish_steepest_descent(opt_pts, max_iter=100, time_limit=deadline - 5)
        except Exception:
            polished_pts, polished_energy = opt_pts, opt_energy
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_points = polished_pts.copy()
    
    # If we didn't get a good result, use the best we have
    if best_points is None:
        best_points = fibonacci_sphere(N)
    
    # Final normalization and validity check
    best_points = project_to_sphere(best_points)
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(N)
        best_points = project_to_sphere(best_points)
    
    # Check minimum distance
    dist_sq = np.sum(best_points**2, axis=1)[:, None] + np.sum(best_points**2, axis=1)[None, :] - 2 * best_points @ best_points.T
    dist_sq = np.clip(dist_sq, 0, None)
    min_dist = np.sqrt(np.min(dist_sq[np.triu_indices(N, k=1)]))
    
    if min_dist < 1e-9:
        # Fallback to fibonacci
        best_points = fibonacci_sphere(N)
        best_points = project_to_sphere(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
