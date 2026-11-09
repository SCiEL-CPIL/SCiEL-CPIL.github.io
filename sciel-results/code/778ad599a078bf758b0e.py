import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the true covering radius in degrees using the convex hull.
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

def _lse_objective_and_grad(points, beta):
    """
    Compute the log-sum-exp of the angular distance to the nearest point
    and its gradient with respect to the point positions.
    
    Objective: (1/beta) * log(sum_j exp(beta * d_j))
    where d_j is the angular distance from point j to its nearest neighbor.
    This approximates the covering radius (max d_j).
    """
    # 1. Compute pairwise angular distances
    # d_ij = arccos(clip(p_i . p_j, -1, 1))
    dots = np.dot(points, points.T)
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # 2. Find nearest neighbor distance for each point
    # d_j = min_{i != j} d_ij
    # Since diagonal is 0 (distance to self), we need to exclude it.
    # We set diagonal to a large value temporarily for min calculation.
    dists_for_min = dists.copy()
    np.fill_diagonal(dists_for_min, np.inf)
    
    nearest_dists = np.min(dists_for_min, axis=1)
    
    # 3. Log-Sum-Exp
    max_d = np.max(nearest_dists)
    if max_d < 1e-10:
        return 0.0, np.zeros_like(points)
        
    # Stability: subtract max
    shifted = beta * (nearest_dists - max_d)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    lse_val = max_d + np.log(sum_exp) / beta
    
    # Weights for the soft-max
    # w_j = exp(beta * d_j) / sum(exp(beta * d_k))
    weights = exp_shifted / sum_exp
    
    # 4. Gradient
    # We want grad of LSE w.r.t. points.
    # LSE = (1/beta) log(sum_j exp(beta d_j))
    # d LSE / d p_k = (1/beta) * (1/sum) * sum_j (exp(beta d_j) * beta * (d d_j / d p_k))
    #               = sum_j (weights[j] * (d d_j / d p_k))
    
    # d_j is the distance to the nearest neighbor of j.
    # Let i* be the index of the nearest neighbor of j.
    # d_j = arccos(p_j . p_{i*})
    # Note: if there are ties, the gradient is undefined, but measure zero.
    # We pick the first argmin.
    
    nn_indices = np.argmin(dists_for_min, axis=1)
    
    # d d_j / d p_j = - p_{i*} / sqrt(1 - (p_j . p_{i*})^2)
    # d d_j / d p_{i*} = - p_j / sqrt(1 - (p_j . p_{i*})^2)
    
    # Note: d_j depends on p_j and p_{i*}.
    # So the gradient contribution to p_k comes from:
    # 1. If k is a point j (the point itself): weight w_j * d d_j / d p_j
    # 2. If k is a point i* (the nearest neighbor of some j): weight w_j * d d_j / d p_{i*}
    
    # Let's compute the gradient vector for each point j (call it g_j)
    # g_j = w_j * (d d_j / d p_j)
    # And scatter the contributions to the neighbors.
    
    # Term 1: Gradient w.r.t. p_j
    # d d_j / d p_j = - p_{i*} / sin(d_j)
    # sin(d_j) = sqrt(1 - cos^2(d_j))
    
    # We need to be careful with numerical stability near 0 or pi.
    # Since it's a covering problem, distances are likely not 0 or pi.
    
    cos_nn = dots[np.arange(N), nn_indices]
    sin_nn = np.sqrt(np.clip(1.0 - cos_nn**2, 1e-15, 1.0))
    
    # d d_j / d p_j
    # The gradient of arccos(x) is -1/sqrt(1-x^2) * dx/dp
    # x = p_j . p_{i*}
    # dx/dp_j = p_{i*}
    # So d d_j / d p_j = - p_{i*} / sin(d_j)
    
    nn_pts = points[nn_indices]
    
    # grad_self[j] = - nn_pts[j] / sin_nn[j]
    # Multiply by weight w_j
    grad_self = -nn_pts * (weights / sin_nn)[:, np.newaxis]
    
    # Term 2: Gradient w.r.t. p_{i*}
    # d d_j / d p_{i*} = - p_j / sin(d_j)
    # Contribution to grad[k] from j where nn_indices[j] == k:
    # grad[k] += w_j * (- p_j / sin(d_j))
    
    grad = np.zeros_like(points)
    
    # Add self contributions
    grad += grad_self
    
    # Add neighbor contributions (scatter)
    # For each j, add w_j * (- p_j / sin_nn[j]) to grad[nn_indices[j]]
    # This can be done with np.add.at or a loop (N=130 is small)
    # Using a loop for clarity and speed (130 iterations is trivial)
    for j in range(N):
        k = nn_indices[j]
        # Contribution to k from j
        # d d_j / d p_k = - p_j / sin_nn[j]
        grad[k] += -weights[j] * points[j] / sin_nn[j]
        
    return lse_val * 180.0 / np.pi, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Riemannian Gradient Descent
    # We use a simple adaptive learning rate.
    # The objective is in degrees, but gradients are in radians effectively (scaled).
    # Let's work in radians for the optimizer to keep numbers small, then convert for scoring.
    # Actually, _lse_objective_and_grad returns value in degrees.
    # Let's modify it to return radians or just scale lr.
    # The derivative of arccos is in radians. The LSE is in radians.
    # The function multiplies by 180/pi at the end.
    # So the gradient returned is in "degree units".
    # To move points on the sphere (unit vectors), the step size should be small.
    
    lr = 0.05
    beta = 20.0
    momentum = 0.0
    v = np.zeros_like(points)
    
    # Adaptive Beta: start low (smooth) and increase (sharper)
    # This helps escape local minima early and polish later.
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # Evaluate objective and gradient
        val_deg, grad = _lse_objective_and_grad(points, beta)
        val_rad = val_deg * np.pi / 180.0
        
        # Momentum (Polyak)
        # v = momentum * v - lr * grad
        # But we need to project v onto the tangent space?
        # Standard approach:
        # 1. Compute gradient in ambient space
        # 2. Project gradient to tangent space: grad_tan = grad - (grad . p) p
        # 3. Update v in tangent space
        # 4. Update p via exp map: p_new = exp_p(v)
        
        # Project gradient to tangent space
        grad_proj = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Momentum update in tangent space
        v = momentum * v - lr * grad_proj
        
        # Optional: project v to tangent space again to remove drift
        v = v - np.sum(v * points, axis=1, keepdims=True) * points
        
        # Exponential map:
        # exp_p(v) = p * cos(||v||) + (v / ||v||) * sin(||v||)
        # For small v, exp_p(v) ≈ p + v
        
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        # Avoid division by zero
        v_norms_safe = np.where(v_norms > 1e-10, v_norms, 1.0)
        
        cos_v = np.cos(v_norms_safe)
        sin_v = np.sin(v_norms_safe)
        
        # Handle zero norm v
        cos_v[v_norms < 1e-10] = 1.0
        sin_v[v_norms < 1e-10] = 0.0
        
        v_dir = v / v_norms_safe
        
        points_new = points * cos_v + v_dir * sin_v
        
        # Renormalize to be safe against numerical drift
        points_new /= np.linalg.norm(points_new, axis=1, keepdims=True)
        
        # Check for improvement
        # We don't want to evaluate the true covering radius (ConvexHull) every step as it's slow.
        # But we can evaluate it periodically.
        
        if iters % 100 == 0:
            current_radius = _covering_radius(points_new)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points_new.copy()
                
                # Adaptive strategy:
                # If we are making good progress, keep going.
                # If we are stuck, maybe change beta or lr.
                if best_radius < 11.5:
                    # Polish phase
                    lr = max(0.001, lr * 0.95)
                    beta = min(100.0, beta * 1.05)
                else:
                    # Exploration phase
                    # Occasionally restart or perturb if stuck?
                    # For now, just adjust beta to smooth out landscape
                    if iters % 500 == 0:
                         beta = max(5.0, beta * 0.8)
                         lr = min(0.1, lr * 1.1)
            
            # Update points to new ones regardless of improvement to maintain flow,
            # but if it got significantly worse, we might want to revert?
            # Gradient descent usually monotonically decreases the surrogate.
            # Let's check the surrogate value.
            new_val_deg, _ = _lse_objective_and_grad(points_new, beta)
            if new_val_deg > val_deg + 1e-3:
                # Surrogate increased, step was too big?
                # Halve lr and retry? Or just revert.
                # For simplicity, revert and reduce lr.
                lr = lr * 0.8
                # Don't update points
            else:
                points = points_new
        
        else:
            # Just update points, assume descent
            points = points_new
            
        # Update momentum
        momentum = min(0.9, momentum + 0.001)
        
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
