import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _soft_max_circumradius(points, beta):
    """
    Computes a smooth approximation of the maximum circumradius.
    Returns (value, gradient)
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        M = len(facets)
        
        # Preallocate
        radii = np.zeros(M)
        c = np.zeros((M, 3))
        valid = np.zeros(M, dtype=bool)
        
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 180.0, np.zeros_like(points)
            
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        
        # Soft max: (1/beta) * log(sum(exp(beta * theta)))
        # We want to minimize the max theta.
        # Let's use log-sum-exp for stability.
        max_theta = np.max(theta)
        if max_theta > 1e-10:
            exp_terms = np.exp(beta * (theta - max_theta))
            log_sum_exp = max_theta + np.log(np.sum(exp_terms)) / beta
        else:
            log_sum_exp = 0.0
            
        # Gradient of log-sum-exp w.r.t. theta
        # d/dtheta (1/beta) * log(sum(exp(beta*theta))) = exp(beta*theta) / sum(exp(beta*theta))
        weights = exp_terms / np.sum(exp_terms) if max_theta > 1e-10 else np.ones(M)/M
        
        # Gradient w.r.t. points
        # d theta / d p_i
        # theta = arccos( p1 . c )
        # c is normalized circumcenter.
        # d theta / d c = -p1 / sqrt(1 - (p1.c)^2)
        # d c / d p_k involves the change in the normal.
        
        # This gradient is complex to derive analytically for all vertices of the facet.
        # Instead, we can use numerical differentiation or a simpler heuristic gradient.
        # Given the complexity, let's use a finite difference approximation for the gradient
        # or a simpler "repulsion from circumcenter" heuristic that approximates the gradient.
        
        # Heuristic Gradient:
        # For each facet, the "force" on its vertices is proportional to the weight
        # and the direction that reduces the circumradius.
        # Moving vertices towards the circumcenter reduces the triangle size and thus the radius.
        # So, gradient contribution to point p_k is -w * (c - p_k).
        
        grad = np.zeros_like(points)
        for k in range(3):
            pts_k = points[facets[:, k]]
            # w_k is the weight for the facet
            # Direction: towards circumcenter
            # We want to minimize radius, so we move points towards c.
            # The "gradient" of the objective (which we want to minimize) points in the direction of increase.
            # So the descent direction is -grad.
            # If we define the objective as the soft-max radius, and we want to minimize it,
            # the gradient of the radius w.r.t. position points away from the circumcenter (increasing radius).
            # So we should add w * (p_k - c) to the gradient (which is then subtracted in the update).
            
            # Actually, let's just use the repulsion force directly as the negative gradient.
            # Force on p_k from facet m: w_m * (p_k - c_m)
            # This pushes points away from the circumcenter of large facets.
            
            # Vectorized addition:
            # We need to scatter the forces.
            # For each facet m, add w_m * (p_k - c_m) to grad[facet[m, k]]
            
            # This loop is O(M). M is ~2N. N=130. So 260 iterations. Fast.
            for m in range(M):
                if valid[m]:
                    idx = facets[m, k]
                    # grad[idx] += weights[m] * (pts_k[m] - c[m])
                    grad[idx] += weights[m] * (pts_k[m] - c[m])
                    
        return log_sum_exp * 180.0 / np.pi, grad
        
    except Exception:
        return 180.0, np.zeros_like(points)

def _objective_and_grad(x):
    """
    Objective function and gradient for L-BFGS-B.
    x is a flattened array of points (N, 3).
    """
    points = x.reshape(N, 3)
    
    # Renormalize to handle any drift, though L-BFGS-B on sphere is tricky.
    # We will use a projected gradient approach manually or just let L-BFGS-B work on the sphere
    # by penalizing deviation from unit norm? 
    # Actually, L-BFGS-B is for box constraints. 
    # We can use a penalty term for norm deviation.
    
    norms = np.linalg.norm(points, axis=1)
    norm_penalty = 100.0 * np.sum((norms - 1.0)**2)
    
    val, grad = _soft_max_circumradius(points, beta=20.0)
    
    # Add gradient of penalty
    # d/dx (||x||^2 - 1)^2 = 4(||x||^2 - 1) * x / ||x|| ? 
    # Simpler: just use the soft-max gradient and re-normalize in a wrapper?
    # L-BFGS-B doesn't support custom projections.
    
    # Alternative: Use a different optimizer that supports projections, or 
    # implement a projected gradient step manually.
    
    # Given the time, let's stick to a simpler, more robust approach:
    # Use the soft-max value as a guide for a custom projected gradient step.
    
    return val + norm_penalty, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # We will use a custom projected gradient descent with the soft-max surrogate.
    # This is more stable than L-BFGS-B on the sphere without a manifold library.
    
    lr = 0.05
    beta = 10.0 # Temperature for soft-max
    
    iterations = 5000
    
    for iter in range(iterations):
        if time.time() > deadline:
            break
            
        # Evaluate soft-max objective and gradient
        val, grad = _soft_max_circumradius(points, beta)
        
        # Add a small repulsion term to ensure global spread and avoid local minima
        # where points cluster.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = np.eye(N, dtype=bool)
        np.fill_diagonal(dists, 1.0)
        forces_global = np.sum(diff / (dists[:, :, np.newaxis] ** 3), axis=1)
        
        # Combine gradients
        # The soft-max gradient points in the direction of increasing radius.
        # We want to minimize, so we move opposite to it.
        # The repulsion force points away from neighbors, which helps spread.
        # We add a scaled repulsion force.
        
        total_grad = grad + 0.1 * forces_global
        
        # Projected Gradient Step
        # Move in direction of -total_grad
        points = points - lr * total_grad
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Periodic evaluation of true radius
        if iter % 50 == 0:
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
                # If we are close to the known best, try to polish with smaller lr
                if best_radius < 11.5:
                    lr = max(0.001, lr * 0.9)
                    beta = min(50.0, beta * 1.1) # Sharper max

    # Final check
    final_radius = _covering_radius(points)
    if final_radius < best_radius:
        best_radius = final_radius
        best_points = points.copy()
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
