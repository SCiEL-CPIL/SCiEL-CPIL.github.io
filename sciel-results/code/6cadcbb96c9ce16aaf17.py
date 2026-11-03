import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        M = len(facets)
        max_radius = 0.0
        
        for k in range(M):
            tri = facets[k]
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
        return np.degrees(max_radius)
    except Exception:
        return 180.0

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin from 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    # Using a soft-max potential field approach
    # Objective: minimize max_j d(i, j) for all i? No, covering radius is max over sphere.
    # Approximation: The covering radius is determined by the vertices of the convex hull facets (circumcenters).
    # We can approximate the "hole" energy by looking at the points themselves.
    # A common heuristic is to minimize the sum of exp(beta * d_ij) or similar.
    # Or, more directly, we can use the gradient of the soft-max of distances to nearest neighbors?
    # Actually, the directive suggests: "Model the covering problem as a continuous potential field where the objective is to minimize the maximum value of the distance to the nearest point, approximated by a differentiable soft-max of pairwise distances".
    
    # Let's define the objective F(points) = sum_i soft_max_j d(i,j)
    # Where d(i,j) is the angular distance between point i and point j.
    # Minimizing this should encourage uniform distribution.
    
    # We will use gradient descent on this objective.
    # We need to compute the gradient of F with respect to points.
    
    # Let D be the matrix of pairwise angular distances.
    # F = sum_i (1/beta) * log(sum_j exp(beta * D[i,j]))
    # dF/dP_k = sum_i d(soft_max_j D[i,j])/dP_k
    
    # This might be computationally expensive to compute analytically.
    # Instead, let's try a simpler potential:
    # Energy = sum_{i,j} exp(-alpha * d_ij) ? No, that's for clustering.
    # We want to push points apart.
    # Energy = sum_{i,j} (1/d_ij)^p ?
    
    # Let's stick to the "soft-max of pairwise distances" idea.
    # For each point i, let m_i = max_j d(i,j).
    # We want to minimize sum_i m_i.
    # Soft max: s_i = (1/beta) log(sum_j exp(beta * d(i,j)))
    # Total energy E = sum_i s_i
    
    # Gradient of E w.r.t. P_k:
    # dE/dP_k = sum_i ds_i/dP_k
    # ds_i/dP_k = (1/beta) * sum_j exp(beta * d_ij) * beta * d(d_ij)/dP_k / sum_j exp(beta * d_ij)
    #           = sum_j w_ij * d(d_ij)/dP_k
    # where w_ij = exp(beta * d_ij) / sum_j exp(beta * d_ij)
    
    # d(d_ij)/dP_k:
    # d_ij = arccos(P_i . P_j)
    # d(d_ij)/dP_k = -1/sqrt(1 - (P_i . P_j)^2) * (d(P_i . P_j)/dP_k)
    # If k == i: d(P_i . P_j)/dP_i = P_j
    # If k == j: d(P_i . P_j)/dP_j = P_i
    # If k != i and k != j: 0
    
    # So d(d_ij)/dP_k is non-zero only if k==i or k==j.
    # This means the gradient for each point P_k depends only on its interactions with others.
    # Gradient for P_k:
    # G_k = sum_{j != k} w_kj * d(d_kj)/dP_k + sum_{i != k} w_ik * d(d_ik)/dP_k
    # Note: d_ij = d_ji, so w_ij = w_ji.
    # Let's check the term d(d_kj)/dP_k.
    # d_kj = arccos(P_k . P_j)
    # d(d_kj)/dP_k = -P_j / sqrt(1 - (P_k . P_j)^2)
    # Similarly d(d_ik)/dP_k = -P_i / sqrt(1 - (P_i . P_k)^2)
    
    # So G_k = - sum_{j != k} w_kj * P_j / sqrt(1 - (P_k . P_j)^2) - sum_{i != k} w_ik * P_i / sqrt(1 - (P_i . P_k)^2)
    # Since w_kj = w_jk, and the sums are over the same set of neighbors, we can combine them?
    # Wait, the first sum is over j (neighbors of k), the second is over i (neighbors of k).
    # They are identical sums.
    # So G_k = -2 * sum_{j != k} w_kj * P_j / sqrt(1 - (P_k . P_j)^2)
    
    # We project this gradient onto the tangent plane of the sphere at P_k.
    # T_k = G_k - (G_k . P_k) P_k
    
    # Then update: P_k <- P_k - step * T_k
    
    # Let's implement this.
    
    beta = 10.0  # Temperature parameter
    step_size = 0.01
    min_step_size = 1e-6
    stagnant_count = 0
    max_stagnant = 20
    
    while time.time() < deadline:
        # Compute pairwise dot products
        dots = points @ points.T
        np.fill_diagonal(dots, 1.0)
        dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        
        # Compute distances
        dists = np.arccos(dots)
        
        # Compute weights w_ij = exp(beta * d_ij) / sum_j exp(beta * d_ij)
        # For numerical stability, subtract max
        max_dists = np.max(dists, axis=1, keepdims=True)
        exp_dists = np.exp(beta * (dists - max_dists))
        sums = np.sum(exp_dists, axis=1, keepdims=True)
        weights = exp_dists / sums
        
        # Set diagonal to 0 (no self-interaction)
        np.fill_diagonal(weights, 0.0)
        
        # Compute gradient
        # G_k = -2 * sum_{j != k} w_kj * P_j / sqrt(1 - (P_k . P_j)^2)
        # Let denom_kj = sqrt(1 - dots_kj^2)
        denom = np.sqrt(1.0 - dots * dots)
        # Avoid division by zero
        denom[denom < 1e-12] = 1e-12
        
        # weighted_points_kj = weights_kj / denom_kj
        # G_k = -2 * sum_j weighted_points_kj * P_j
        # This is a matrix multiplication:
        # G = -2 * ( (weights / denom) @ points )
        
        coeff = weights / denom
        gradients = -2.0 * (coeff @ points)
        
        # Project to tangent plane
        dot_grad = np.sum(gradients * points, axis=1, keepdims=True)
        tangent_grads = gradients - dot_grad * points
        
        # Update
        new_points = points - step_size * tangent_grads
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        if not np.all(np.isfinite(new_points)):
            break
            
        points = new_points
        
        # Evaluate radius
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            stagnant_count = 0
        else:
            stagnant_count += 1
            step_size = max(step_size * 0.95, min_step_size)
            
            if stagnant_count > max_stagnant:
                # Random kick
                points += rng.normal(0, 0.05, points.shape)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                stagnant_count = 0
                
                # Also try to increase beta to focus on the max
                beta *= 1.2
                if beta > 100.0:
                    beta = 100.0

    return best_points
