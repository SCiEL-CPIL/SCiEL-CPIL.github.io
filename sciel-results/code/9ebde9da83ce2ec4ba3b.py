import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s kill

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on a unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _covering_radius_hull(points):
    """
    Compute the covering radius in radians using the convex hull.
    The covering radius is the maximum over all facets of the angular distance
    from the facet's circumcenter (on the sphere) to its vertices.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback: worst case (hemisphere)
    
    max_angle = 0.0
    # Iterate over all simplices (facets) of the convex hull
    for simplex in hull.simplices:
        # Extract vertices
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the normal to the plane containing the triangle
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        # Ensure the normal points outward (away from origin)
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        # Calculate angular distance from circumcenter (normal) to vertices
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        # The radius of the cap is the max of these distances
        facet_radius = max(d1, d2, d3)
        
        # Check if this cap is empty.
        # A cap is empty if no other point is closer to the circumcenter than the vertices.
        # Distance from circumcenter to any point v is arccos(C . v).
        # We need to ensure min_{k not in simplex} arccos(C . v_k) >= facet_radius.
        # Equivalent to max_{k not in simplex} C . v_k <= cos(facet_radius).
        
        # For N=130, checking all points for each facet is O(N^2) which is fine (130^2 = 16900).
        dots = points @ normal
        # Mask out the vertices
        mask = np.ones(N, dtype=bool)
        mask[simplex] = False
        max_dot_other = np.max(dots[mask])
        
        # If max_dot_other > cos(facet_radius), then there is a point closer than the vertices.
        # In that case, the circumcenter is not the local maximum of the min-distance function.
        # So we skip this facet.
        if max_dot_other > np.cos(facet_radius) + 1e-12:
            continue
            
        max_angle = max(max_angle, facet_radius)
        
    return max_angle

def _covering_radius_grid(points, n_theta=360, n_phi=720):
    """
    Compute the covering radius in radians using a dense spherical grid.
    This is a differentiable-friendly approximation (though we use it for evaluation here).
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    
    T, P = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find the nearest configuration point
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid_points), chunk_size):
        end = min(start + chunk_size, len(grid_points))
        chunk = grid_points[start:end]
        
        # Compute distances (dot products)
        dots = chunk @ points.T
        # Clip to [-1, 1]
        dots = np.clip(dots, -1.0, 1.0)
        # Max dot product gives min angular distance
        max_dots = dots.max(axis=1)
        # Angular distance = arccos(max_dot)
        angles = np.arccos(max_dots)
        if angles.max() > max_dist:
            max_dist = angles.max()
            
    return max_dist

def _repulsion_optimize(points, max_iters=1000, lr=0.01, deadline=None):
    """
    Optimize points on sphere using repulsion from nearest neighbors.
    This is a heuristic to spread points out.
    """
    points = points.copy().astype(float)
    n = len(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise dot products
        dots = np.clip(points @ points.T, -1, 1)
        angles = np.arccos(dots)
        np.fill_diagonal(angles, 0)
        
        # Repulsion: push points away from nearby points
        # Use a kernel that decays with distance
        # Force is sum of vectors away from neighbors, weighted by distance
        # Only consider neighbors within a certain angle to save computation?
        # For N=130, full matrix is fine.
        
        # We want to maximize the minimum distance.
        # A common heuristic is to move points in the direction of the gradient of the sum of repulsive potentials.
        # Potential V = sum_{i<j} f(d_ij).
        # To maximize min distance, we can minimize sum of 1/d_ij or similar.
        # Let's use a simple force: f_i = sum_{j != i} (p_i - p_j) / (1 + d_ij^2)
        # This pushes p_i away from p_j.
        
        # Vectorized force calculation
        # p_i - p_j is (points[:, None, :] - points[None, :, :])
        # We want to sum over j for each i.
        
        # To avoid O(N^2) memory if N was large, but N=130 is small.
        diff = points[:, None, :] - points[None, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=2)  # (N, N)
        
        # Weight: 1 / (1 + dist_sq)
        # Set diagonal to 0
        weights = 1.0 / (1.0 + dist_sq)
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum_j weights[i,j] * diff[i,j]
        forces = np.sum(weights[:, :, None] * diff, axis=1)  # (N, 3)
        
        # Project force onto tangent plane
        # radial = (forces . p_i) * p_i
        radial = (forces * points).sum(axis=1, keepdims=True) * points
        tangent_forces = forces - radial
        
        # Update
        points = points + lr * tangent_forces
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    return points

def _gradient_optimize(points, deadline=None):
    """
    Use L-BFGS-B to minimize a smooth approximation of the covering radius.
    We minimize the log-sum-exp of the distances to a dense grid, which is a smooth upper bound.
    """
    # Generate a grid for the objective function
    n_theta = 180
    n_phi = 360
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Beta for the soft-max (log-sum-exp) approximation
    beta = 20.0
    
    def objective_and_grad(flat_params):
        # Reshape to (N, 3)
        p = flat_params.reshape(N, 3)
        
        # Normalize to ensure they are on the sphere (helps with optimization stability)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        
        # Compute dot products with grid
        # dots: (M, N)
        dots = grid_points @ p.T
        dots = np.clip(dots, -1.0, 1.0)
        
        # For each grid point, the distance to the nearest point is arccos(max_dot).
        # We want to maximize the minimum distance, i.e., minimize the maximum distance.
        # Let's use the log-sum-exp trick on the exponentiated distances or something similar.
        # Actually, minimizing the sum of exp(beta * d_i) where d_i is the distance to nearest point.
        # d_i = arccos(max_k dot(g_i, p_k))
        
        max_dots = dots.max(axis=1)
        distances = np.arccos(max_dots)
        
        # Objective: sum(exp(beta * distances))
        # This is a smooth approximation to max(distances)
        obj = np.sum(np.exp(beta * distances))
        
        # Gradient
        # d_obj/d_p_k = sum_i exp(beta * d_i) * beta * d_d_i/d_p_k
        # d_d_i/d_p_k = d_arccos/d_max_dot * d_max_dot/d_p_k
        # d_arccos(x)/dx = -1/sqrt(1-x^2)
        # d_max_dot/d_p_k is 1 if p_k is the nearest, 0 otherwise (subgradient).
        # For a smooth approximation, we can use the softmax over the dot products?
        # Let's use the gradient of the log-sum-exp of the distances directly?
        # Or simpler: use the fact that the gradient of the max is the gradient of the argmax.
        # For a smoother version, we can use the gradient of the sum of exp(beta * d_i).
        
        # Let's compute the gradient numerically? No, analytic is better.
        # Let's stick to a simpler heuristic gradient or use scipy's numerical gradient if analytic is complex.
        # However, for L-BFGS-B, we can provide the gradient.
        
        # Let's try to compute the gradient analytically.
        # Let M_i = max_k (g_i . p_k).
        # d_i = arccos(M_i).
        # f = sum_i exp(beta * d_i).
        # df/dp_k = sum_i exp(beta * d_i) * beta * (-1/sqrt(1-M_i^2)) * (g_i if k is argmax else 0)
        
        # To make it smoother, we can replace the hard argmax with a softmax over the dot products.
        # Let weights w_ik = exp(gamma * (g_i . p_k)) / sum_j exp(gamma * (g_i . p_j))
        # Then M_i is approximated by sum_k w_ik (g_i . p_k).
        # This is a convex combination of the dot products, so it's differentiable.
        
        gamma = 10.0
        # Compute weights
        exp_dots = np.exp(gamma * dots)
        sum_exp_dots = exp_dots.sum(axis=1, keepdims=True)
        weights = exp_dots / sum_exp_dots  # (M, N)
        
        # M_i = sum_k weights[i,k] * dots[i,k]
        M_i = (weights * dots).sum(axis=1)
        M_i = np.clip(M_i, -1.0, 1.0)
        
        # d_i = arccos(M_i)
        # To avoid issues with sqrt(0), clip M_i
        M_i_safe = np.clip(M_i, -0.999999, 0.999999)
        d_i = np.arccos(M_i_safe)
        
        # f = sum_i exp(beta * d_i)
        obj = np.sum(np.exp(beta * d_i))
        
        # Gradient
        # df/dp_k = sum_i exp(beta * d_i) * beta * (-1/sqrt(1-M_i^2)) * d_M_i/dp_k
        # d_M_i/dp_k = weights[i,k] * g_i + dots[i,k] * d_weights[i,k]/dp_k
        # This gets complicated. Let's approximate the gradient of the objective with respect to p_k
        # by just considering the term where p_k contributes to M_i.
        
        # Actually, let's use a simpler objective: sum of squared distances? No, that minimizes mean.
        # Let's use the gradient of the log-sum-exp of the distances.
        # log(sum(exp(beta * d_i)))
        # grad = (1/sum(exp(beta * d_i))) * sum(exp(beta * d_i) * beta * grad_d_i)
        
        # Let's just use numerical gradient for simplicity and robustness, or a very simple analytic one.
        # Given the complexity, let's use a heuristic gradient:
        # Move points away from the grid points that are far from any point.
        
        # Far grid points: those with large d_i.
        # weight_i = exp(beta * d_i) / sum(exp(beta * d_i))
        
        exp_d = np.exp(beta * d_i)
        sum_exp_d = exp_d.sum()
        if sum_exp_d == 0:
            return obj, np.zeros_like(flat_params)
        w_i = exp_d / sum_exp_d  # (M,)
        
        # For each grid point i, it wants to be closer to some point k.
        # The point k that is currently nearest (or has high weight in softmax) should move towards i?
        # No, we want to minimize the distance. So we want to move p_k towards g_i.
        # Force on p_k = sum_i w_i * weights[i,k] * (g_i - p_k)
        
        # weights[i,k] is the softmax weight of p_k for grid point i.
        # force_k = sum_i w_i * weights[i,k] * (g_i - p_k)
        
        # Vectorized:
        # w_i: (M,)
        # weights: (M, N)
        # g_i: (M, 3)
        # p_k: (N, 3)
        
        # force_k = sum_i (w_i * weights[i,k]) * g_i - (sum_i w_i * weights[i,k]) * p_k
        # Let W_k = sum_i w_i * weights[i,k]
        # Let G_k = sum_i w_i * weights[i,k] * g_i
        
        # W: (N,) = weights.T @ w_i
        W = weights.T @ w_i
        # G: (N, 3) = (weights.T @ (w_i[:,None] * grid_points))
        # w_i[:,None] * grid_points: (M, 3)
        G = weights.T @ (w_i[:, None] * grid_points)
        
        forces = G - W[:, None] * p
        
        # Project to tangent plane
        radial = (forces * p).sum(axis=1, keepdims=True) * p
        tangent_forces = forces - radial
        
        # Gradient of the objective with respect to p is -forces (since we are minimizing distance)
        # Wait, if we move p towards g, distance decreases.
        # So the gradient of the distance is in the direction (p - g).
        # We want to minimize the objective, so we move in the direction -gradient.
        # Gradient of objective wrt p is roughly -forces.
        # So we return -tangent_forces as the gradient?
        # Let's check signs.
        # If p is far from g, force is towards g. We want to move p towards g.
        # Gradient descent: p_new = p - lr * grad.
        # So grad should be -force.
        
        grad = -tangent_forces
        
        # Flatten gradient
        grad_flat = grad.ravel()
        
        return obj, grad_flat
    
    # Initial parameters
    x0 = points.ravel()
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-5, 'maxfun': 200}
        )
        if res.success or res.fun < objective_and_grad(x0)[0]:
            new_points = res.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            return new_points
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial configurations
    inits = []
    # Fibonacci
    inits.append(_fibonacci_sphere(N, offset=0.5))
    inits.append(_fibonacci_sphere(N, offset=0.1))
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_points = None
    best_radius = np.pi
    
    # Evaluate initial configurations
    for pts in inits:
        r = _covering_radius_hull(pts)
        if r < best_radius:
            best_radius = r
            best_points = pts.copy()
        if time.time() > deadline - 10:
            break
            
    # Optimize from best initial config using repulsion
    if best_points is not None and time.time() < deadline - 5:
        opt_pts = _repulsion_optimize(best_points, max_iters=1000, lr=0.005, deadline=deadline - 5)
        r = _covering_radius_hull(opt_pts)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
            
    # Try gradient optimization from the best point found so far
    if best_points is not None and time.time() < deadline - 10:
        grad_pts = _gradient_optimize(best_points, deadline=deadline - 5)
        r = _covering_radius_hull(grad_pts)
        if r < best_radius:
            best_radius = r
            best_points = grad_pts.copy()
            
    # Try optimizing from other initial configs if time permits
    for pts in inits:
        if time.time() > deadline - 10:
            break
        # Skip if it's the one we already optimized
        if np.allclose(pts, best_points):
            continue
            
        opt_pts = _repulsion_optimize(pts, max_iters=500, lr=0.005, deadline=deadline - 5)
        r = _covering_radius_hull(opt_pts)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
            
            # If we found a new best, try gradient on it
            if time.time() < deadline - 10:
                grad_pts = _gradient_optimize(best_points, deadline=deadline - 5)
                r = _covering_radius_hull(grad_pts)
                if r < best_radius:
                    best_radius = r
                    best_points = grad_pts.copy()
            
    # Final refinement: Local search / Hill climbing on the best solution
    # We can do a few steps of gradient descent on the covering radius directly?
    # Or just more repulsion steps with smaller lr?
    if best_points is not None and time.time() < deadline - 2:
        # Fine-tune with smaller learning rate
        opt_pts = _repulsion_optimize(best_points, max_iters=200, lr=0.001, deadline=deadline - 1)
        r = _covering_radius_hull(opt_pts)
        if r < best_radius:
            best_points = opt_pts.copy()
            
        # One more gradient step?
        if time.time() < deadline - 1:
            grad_pts = _gradient_optimize(best_points, deadline=deadline - 0.5)
            r = _covering_radius_hull(grad_pts)
            if r < best_radius:
                best_points = grad_pts.copy()

    # Ensure output is valid
    if best_points is None:
        best_points = inits[0]
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    r = _covering_radius_hull(pts)
    print(f"Covering radius: {np.degrees(r):.6f} degrees")
