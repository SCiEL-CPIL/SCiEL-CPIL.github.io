import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        # Vectorized calculation of circumradii
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross, axis=1)
        
        # Avoid division by zero
        valid = norm_cross > 1e-12
        if not np.any(valid):
            return 0.0
            
        c = np.zeros_like(cross)
        c[valid] = cross[valid] / norm_cross[valid, np.newaxis]
        
        # Ensure c points outward (same direction as p1)
        dots = np.einsum('ij,ij->i', c, p1)
        flip = dots < 0
        c[flip] = -c[flip]
        
        # Calculate angular distance
        dot_vals = np.einsum('ij,ij->i', c, p1)
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii = np.arccos(dot_vals)
        
        max_radius = np.max(radii)
        
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _facet_data(points):
    """Compute circumcenters and radii for all facets of the convex hull."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-12
    if not np.any(valid):
        return None, None, None
        
    circumcenters = np.zeros((len(facets), 3))
    radii = np.zeros(len(facets))
    
    # Only compute for valid triangles
    c = np.zeros_like(cross)
    c[valid] = cross[valid] / norm_cross[valid, np.newaxis]
    
    # Orient outward
    dots = np.einsum('ij,ij->i', c, p1)
    flip = dots < 0
    c[flip] = -c[flip]
    
    circumcenters = c
    
    dot_vals = np.einsum('ij,ij->i', c, p1)
    dot_vals = np.clip(dot_vals, -1.0, 1.0)
    radii = np.arccos(dot_vals)
    
    return circumcenters, radii, facets

def _smooth_objective_and_grad(x, temperature=0.01):
    """
    Compute smooth max of facet circumradii and its gradient.
    x is flattened array of shape (3*N,).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    points = points / norms
    
    circumcenters, radii, facets = _facet_data(points)
    
    if circumcenters is None:
        return 1.5, np.zeros(3 * N)
    
    if len(radii) == 0:
        return 0.0, np.zeros(3 * N)
    
    # Smooth max
    max_r = np.max(radii)
    exps = np.exp((radii - max_r) / temperature)
    smooth_val = max_r + temperature * np.log(np.sum(exps))
    
    # Weights
    sum_exps = np.sum(exps)
    if sum_exps == 0:
        return smooth_val, np.zeros(3 * N)
        
    weights = exps / sum_exps
    
    # Gradient computation
    grad_flat = np.zeros(3 * N)
    
    # Precompute sin(r) for all facets
    cos_r = np.einsum('ij,ij->i', circumcenters, points[facets[:, 0]]) # Using p1 is fine for cos, as c dot p1 = c dot p2 = c dot p3
    sin_r = np.sqrt(np.maximum(1e-15, 1.0 - cos_r * cos_r))
    
    # grad_dir for each facet: -c / sin(r)
    grad_dirs = -circumcenters / sin_r[:, np.newaxis]
    
    # For each vertex in each facet, add weighted tangent gradient
    # facets shape (M, 3)
    # We need to scatter the gradients
    
    for k in range(3):
        indices = facets[:, k]
        # grad_dirs is (M, 3). weights is (M,).
        # We want to add weights[i] * grad_dirs[i] to grad_flat[indices[i]]
        # But we must project to tangent space first?
        # The gradient of the distance with respect to the point p on the sphere is the tangent vector.
        # grad_dir is already the direction on the sphere?
        # Let's check: d/dp arccos(c.p) = -c / sqrt(1-(c.p)^2).
        # This vector is in R^3. To get the gradient on the manifold, we project to the tangent space.
        # Projection of v onto tangent space at p is v - (v.p)p.
        
        # However, c is perpendicular to the triangle, not necessarily to p.
        # But c is the circumcenter. For a triangle on the unit sphere, the circumcenter c is not necessarily in the plane of the triangle?
        # Wait, the "circumcenter" in the context of the covering radius on the sphere is the point on the sphere equidistant from the 3 vertices.
        # The vector c computed as cross(v1, v2) normalized is the normal to the plane of the triangle.
        # Is this the same as the circumcenter on the sphere?
        # Yes, the point on the sphere equidistant from p1, p2, p3 lies on the line passing through the origin and the normal of the plane p1p2p3.
        # So c is the correct direction.
        
        # The gradient of the angle r = arccos(c.p) with respect to p is -c/sin(r).
        # This vector is in R^3.
        # The constrained gradient (tangent) is P_p (-c/sin(r)) where P_p(v) = v - (v.p)p.
        
        # Let's compute the projection for each vertex.
        
        # We can vectorize this loop over the 3 vertices of the triangles.
        
        pass # The loop below is fine for M ~ 200-400 facets.

    # Vectorized scatter-add is tricky with different indices. Loop is fast enough.
    for idx in range(len(facets)):
        w = weights[idx]
        if w < 1e-8:
            continue
            
        g_dir = grad_dirs[idx]
        
        for v_idx in facets[idx]:
            p = points[v_idx]
            # Project g_dir to tangent space at p
            proj = np.dot(g_dir, p)
            tangent_grad = g_dir - proj * p
            grad_flat[3*v_idx:3*v_idx+3] += w * tangent_grad
            
    # Scale by 1/||x|| to account for normalization in the objective function
    # f(x) = F(x/||x||). Grad_x f = J^T Grad_p F.
    # J = 1/||x|| (I - xx^T/||x||^2).
    # Since Grad_p F is tangent (orthogonal to p=x/||x||), xx^T Grad_p F = 0.
    # So Grad_x f = 1/||x|| Grad_p F.
    
    grad_flat = grad_flat.reshape(N, 3) / norms
    
    return smooth_val, grad_flat.flatten()

def _polish_worst_facets(points, max_iters=50, tol=1e-10):
    """
    Iteratively polish the worst facets by moving their vertices.
    """
    for _ in range(max_iters):
        circumcenters, radii, facets = _facet_data(points)
        if circumcenters is None:
            break
        
        if len(radii) == 0:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        if max_r < tol:
            break
        
        tri = facets[max_idx]
        c = circumcenters[max_idx]
        
        # Move vertices towards the circumcenter to reduce the radius
        # This is a heuristic step.
        for v_idx in tri:
            p = points[v_idx]
            # Direction to move: towards c, projected to tangent space
            direction = c - p
            proj = np.dot(direction, p)
            tangent_dir = direction - proj * p
            norm_td = np.linalg.norm(tangent_dir)
            if norm_td > 1e-12:
                tangent_dir /= norm_td
                # Small step
                points[v_idx] = points[v_idx] + 0.005 * tangent_dir
                points[v_idx] /= np.linalg.norm(points[v_idx])
                
    return points

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: L-BFGS-B optimization with smooth max surrogate
    temperature = 0.05
    stall_count = 0
    max_stall = 5
    
    while time.time() < deadline - 30:
        x0 = points.flatten()
        
        def obj_and_grad(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-5, 'maxls': 20}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            if not np.all(np.isfinite(new_points)):
                break
            
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                temperature *= 0.8
                stall_count = 0
            else:
                temperature *= 0.7
                stall_count += 1
                if temperature < 0.001:
                    break
                    
                # Restart strategy: if stalled, perturb and try again
                if stall_count >= max_stall:
                    # Add small noise
                    noise = np.random.normal(0, 0.01, (N, 3))
                    points = points + noise
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    stall_count = 0
                    max_stall = max(3, max_stall - 1) # Reduce patience
                    
        except Exception:
            temperature *= 0.7
            if temperature < 0.001:
                break
    
    # Phase 2: Polish worst facets
    if time.time() < deadline - 5:
        polished = _polish_worst_facets(best_points.copy(), max_iters=100, tol=1e-12)
        polished_radius = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished.copy()
    
    # Phase 3: Final fine-tuning with very small L-BFGS-B steps
    if time.time() < deadline - 2:
        temperature = 0.001
        x0 = best_points.flatten()
        
        def obj_and_grad_final(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-7, 'maxls': 10}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            if np.all(np.isfinite(new_points)):
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points
