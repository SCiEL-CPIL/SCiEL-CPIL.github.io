"""
Strategy: Non-convex Quadratic Programming via Projected Gradient Ascent.
We reframe the problem as maximizing the minimum pairwise angular separation.
This is equivalent to minimizing the maximum angular distance from any point on the sphere
to its nearest neighbor, under the assumption that the optimal configuration is well-spread
such that the covering radius is determined by the "worst-case" gap between neighbors.

We use a differentiable surrogate objective:
f(P) = sum_i [ -log( sum_j exp(-beta * (P_i . P_j)) ) ]
This is a soft-min of the pairwise dot products. Minimizing this function pushes points
apart to maximize the minimum dot product (and thus minimize the maximum angle).

We perform projected gradient ascent on the sphere:
1. Compute gradient of the surrogate objective.
2. Update points: P_new = P + lr * grad.
3. Project back to unit sphere: P_new = P_new / ||P_new||.
4. Use adaptive learning rate and annealing of beta to refine the solution.

This approach avoids the discrete jumps of Lloyd's algorithm and the geometric complexities
of convex hull repair, relying instead on smooth gradient flow to find a dense, uniform packing.
"""

import numpy as np
import time

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Uses ConvexHull to find facets and computes the circumradius of each facet.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Circumcenter of spherical triangle
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Orient towards centroid
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Angular distance from c to p1
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _soft_min_gradient(points, beta):
    """
    Compute the gradient of the surrogate objective:
    f(P) = sum_i [ -log( sum_j exp(-beta * (P_i . P_j)) ) ]
    
    We want to MINIMIZE f(P), which corresponds to MAXIMIZING the min dot product.
    So we compute grad f and subtract it (gradient descent on f, or ascent on -f).
    
    Let S_i = sum_j exp(-beta * (P_i . P_j))
    f_i = -log(S_i)
    
    df_i / dP_i = - (1/S_i) * dS_i / dP_i
    dS_i / dP_i = sum_j [ exp(-beta * (P_i . P_j)) * (-beta) * d(P_i . P_j)/dP_i ]
    
    Note: d(P_i . P_j)/dP_i = P_j for j != i, and 0 for j == i (since P_i . P_i = 1 is constant on sphere, 
    but in the sum, the j=i term is exp(0)=1, and its derivative w.r.t P_i is 0? 
    Actually, if we treat P_i as a variable in Euclidean space, d(P_i.P_j)/dP_i = P_j.
    The constraint ||P_i||=1 is handled by projection.
    So d(P_i . P_j)/dP_i = P_j for all j.
    
    Wait, the term j=i is exp(-beta * 1) = exp(-beta).
    d/dP_i (P_i . P_i) = 2 P_i.
    So dS_i/dP_i = sum_{j!=i} exp(-beta * P_i.P_j) * (-beta) * P_j + exp(-beta) * (-beta) * 2 P_i
    
    However, usually in these repulsion models, we ignore the self-term or treat it as constant.
    Let's include it properly.
    
    Let's use a vectorized approach.
    D = P @ P.T  (N x N matrix of dot products)
    E = exp(-beta * D)
    S = E.sum(axis=1)  # Sum over j for each i
    
    The gradient contribution to P_i from term f_i is:
    - (1/S_i) * sum_j E_ij * (-beta) * P_j  (for j != i, d/dP_i P_i.P_j = P_j)
    - (1/S_ii) * E_ii * (-beta) * 2 P_i    (for j = i, d/dP_i P_i.P_i = 2 P_i)
    
    Actually, let's simplify. The self-term E_ii = exp(-beta) is constant with respect to the direction 
    if we assume normalization, but in the gradient step before normalization, it varies.
    However, a common simplification is to exclude j=i from the sum to avoid the 2*P_i term complexity 
    or just note that the self-term provides a constant force? No, it's not constant.
    
    Let's stick to the standard repulsion force which is sum_{j!=i} P_j * exp(-beta P_i.P_j).
    This is effectively the gradient of the log-sum-exp of the other points.
    The self-term contributes a force proportional to P_i, which just scales the point. 
    Since we project to the sphere, a force along P_i does not change the direction. 
    Therefore, we can safely ignore the j=i term in the gradient direction!
    
    So, grad f_i ≈ - (1/S_i) * sum_{j!=i} E_ij * (-beta) * P_j
                   = (beta / S_i) * sum_{j!=i} E_ij * P_j
    
    We want to minimize f, so we move in direction -grad f.
    Direction = - (beta / S_i) * sum_{j!=i} E_ij * P_j
    
    Wait, if we minimize f, and f is "soft min", we want to increase the min dot product.
    Increasing dot product means moving P_i closer to its neighbors? No, that's wrong.
    Min dot product is the "closest" neighbor. We want to MAXIMIZE the min dot product.
    So we want to move P_i AWAY from its neighbors.
    
    Let's check the sign.
    f = -log(sum exp(-beta d)). If d increases, exp(-beta d) decreases, sum decreases, log decreases, -log increases.
    So f increases as d increases.
    We want to maximize d (min dot product). So we want to MAXIMIZE f.
    So we should do Gradient ASCENT on f.
    Direction = + grad f.
    
    grad f_i = - (1/S_i) * dS_i/dP_i
    dS_i/dP_i = sum_{j!=i} E_ij * (-beta) * P_j
    grad f_i = - (1/S_i) * (-beta) * sum_{j!=i} E_ij P_j
             = (beta / S_i) * sum_{j!=i} E_ij P_j
    
    So the update is P_i <- P_i + lr * (beta / S_i) * sum_{j!=i} E_ij P_j.
    This moves P_i in the direction of the weighted average of its neighbors.
    Wait, that moves it TOWARDS the neighbors? That would DECREASE the distance.
    
    Let's re-evaluate.
    Objective: Maximize min_{j!=i} (P_i . P_j).
    Surrogate: G = sum_i log( sum_j exp(beta * (P_i . P_j)) )  <-- Soft MAX of dot products?
    If we use Soft Max, we are targeting the MAX dot product, i.e., the CLOSEST neighbor.
    We want to MINIMIZE the max dot product (to push closest neighbors away).
    
    Let's define H = sum_i log( sum_j exp(beta * (P_i . P_j)) ).
    Minimizing H will push the max dot product down.
    
    dH_i/dP_i = (1/S_i) * sum_j E_ij * beta * P_j
    Where S_i = sum_j exp(beta P_i.P_j), E_ij = exp(beta P_i.P_j).
    
    Gradient descent on H:
    P_i <- P_i - lr * (beta / S_i) * sum_{j!=i} E_ij P_j  (ignoring self-term as it's parallel to P_i)
    
    This moves P_i AWAY from the weighted average of its neighbors (weighted by proximity).
    This is the correct repulsive force.
    """
    D = points @ points.T
    # Mask diagonal to avoid self-interaction in the sum for gradient direction
    # But for the normalization S_i, we can include or exclude. 
    # Including self-term: E_ii = exp(beta). It's constant.
    # Let's exclude diagonal from the sum for the force direction to be safe.
    
    E = np.exp(beta * D)
    # Set diagonal to 0 for the force sum
    np.fill_diagonal(E, 0.0)
    
    # S_i should be the sum of the weights. 
    # If we used the log-sum-exp of the full set, S_i includes self.
    # If we ignore self in force, we should be consistent.
    # Let's use S_i = sum_{j!=i} E_ij.
    S = E.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    S[S == 0] = 1e-10
    
    # Force = (beta / S_i) * sum_{j!=i} E_ij P_j
    # This is (beta * (E @ points) / S)
    forces = (beta * (E @ points)) / S
    
    # We want to MINIMIZE H, so we move in direction -forces
    return -forces

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
    
    # Parameters
    beta = 10.0
    lr = 0.01
    iterations = 2000
    
    for iter in range(iterations):
        if time.time() > deadline:
            break
            
        # Compute covering radius periodically (it's expensive)
        if iter % 50 == 0:
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        # Gradient step
        grad = _soft_min_gradient(points, beta)
        
        # Update
        points = points + lr * grad
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Anneal beta to increase precision
        # Start with low beta (smooth), increase to high beta (sharp, focuses on nearest neighbors)
        beta = 10.0 * (1.0 + 0.001 * iter)
        
        # Decay learning rate
        lr = 0.01 * (0.999 ** iter)
        
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
