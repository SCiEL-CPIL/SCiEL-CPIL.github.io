"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumcenters and radii for each triangular facet
            # For a triangle with vertices a, b, c on the unit sphere, 
            # the circumcenter (on the sphere) is proportional to the normal vector of the plane.
            # The radius is the angle between the circumcenter and any vertex.
            
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            norms = np.linalg.norm(normal, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                # Normalize normals to get circumcenter directions
                cc = normal[valid] / norms[valid, np.newaxis]
                # Ensure orientation (pointing outward from origin)
                # The dot product with the centroid of the face should be positive
                centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                dots = np.sum(cc * centroids, axis=1)
                cc[dots < 0] *= -1
                
                # Radius is arccos(dot(cc, a))
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                # The radius is the angular distance. 
                # Note: If the circumcenter is on the opposite side, the "cap" covering the facet 
                # is actually the larger one? No, the convex hull facet is a flat triangle.
                # The "hole" is the spherical triangle. The circumradius is the distance to the circumcenter.
                radii[valid] = r
            
            if np.any(~valid):
                # Degenerate facet, use max edge length as proxy
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            return np.inf

    def make_grid(n_theta=80, n_phi=160):
        """Create a dense grid of points on the unit sphere."""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_points = make_grid(80, 160)  # ~12800 points

    def soft_max_objective(flat_pts, beta=50.0):
        """
        Smooth surrogate for covering radius using soft-max.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        dots = grid_points @ pts.T  # shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # shape (G, N)
        
        # Soft-min over j (config points) for each grid point i
        # We want min_j dists[i,j]. 
        # Soft min is -1/beta * log(sum(exp(-beta * dists)))
        neg_beta_dists = -beta * dists
        max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
        log_sum_exp_inner = np.log(np.sum(np.exp(neg_beta_dists - max_neg), axis=1)) + max_neg.ravel()
        soft_min = -log_sum_exp_inner / beta  # shape (G,)
        
        # Soft-max over grid points i
        # We want max_i soft_min[i].
        # Soft max is 1/beta * log(sum(exp(beta * soft_min)))
        beta_soft_min = beta * soft_min
        max_outer = np.max(beta_soft_min)
        log_sum_exp_outer = np.log(np.sum(np.exp(beta_soft_min - max_outer))) + max_outer
        objective = log_sum_exp_outer / beta
        
        # Gradient computation
        outer_weights = np.exp(beta_soft_min - max_outer)
        outer_weights /= np.sum(outer_weights)  # shape (G,)
        
        inner_weights = np.exp(neg_beta_dists - max_neg)  # shape (G, N)
        inner_weights /= np.sum(inner_weights, axis=1, keepdims=True)  # shape (G, N)
        
        # d dists / d pts = - (grid_points - pts*dot) / sqrt(1-dot^2)
        # This is the gradient of arccos(dot) w.r.t pts.
        # grad_dists = - (grid_points[:, np.newaxis, :] - dots[:, np.newaxis, :] * pts[np.newaxis, :, :]) / sqrt(1-dots^2)
        # But we have inner_weights which is softmax of -dists.
        # d objective / d pts = sum_i outer_weights[i] * sum_j inner_weights[i,j] * d dists[i,j] / d pts[j]
        
        one_minus_dot_sq = 1.0 - dots * dots
        one_minus_dot_sq = np.maximum(one_minus_dot_sq, 1e-12)
        sqrt_oms = np.sqrt(one_minus_dot_sq)
        
        # Gradient of arccos(x) is -1/sqrt(1-x^2). 
        # d/dp arccos(g.p) = - (g - p*(g.p)) / sqrt(1-(g.p)^2)
        # So d dists / d pts[j] = - (grid_points[i] - pts[j]*dots[i,j]) / sqrt(1-dots[i,j]^2)
        
        # weighted_coeff[i,j] = outer_weights[i] * inner_weights[i,j]
        weighted_coeff = outer_weights[:, np.newaxis] * inner_weights  # shape (G, N)
        
        # grad_pts[j] = - sum_i weighted_coeff[i,j] * (grid_points[i] - pts[j]*dots[i,j]) / sqrt_oms[i,j]
        # Let's compute this efficiently.
        # term1 = sum_i weighted_coeff[i,j] * grid_points[i] / sqrt_oms[i,j]
        # term2 = sum_i weighted_coeff[i,j] * pts[j] * dots[i,j] / sqrt_oms[i,j]
        
        coeff_div_sqrt = weighted_coeff / sqrt_oms  # shape (G, N)
        
        # sum over i for each j
        # We need sum_i (coeff_div_sqrt[i,j] * grid_points[i])
        # This is (grid_points.T @ coeff_div_sqrt) -> shape (3, N)
        term1 = grid_points.T @ coeff_div_sqrt  # shape (3, N)
        
        # We need sum_i (coeff_div_sqrt[i,j] * dots[i,j])
        term2_scalar = np.sum(coeff_div_sqrt * dots, axis=0)  # shape (N,)
        
        # grad_pts = - (term1 - term2_scalar[:, np.newaxis] * pts)
        grad_pts = -(term1.T - term2_scalar[:, np.newaxis] * pts)
        
        # Project to tangent space
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
        grad_flat = grad_tangent.ravel()
        
        return objective, grad_flat

    def targeted_refinement(points, max_iters=300):
        """Refine by targeting the worst facet: project vertices toward antipode of circumcenter."""
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                norms = np.linalg.norm(normal, axis=1)
                
                radii = np.zeros(len(faces))
                valid = norms > 1e-10
                
                if np.any(valid):
                    cc = normal[valid] / norms[valid, np.newaxis]
                    centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                    dots = np.sum(cc * centroids, axis=1)
                    cc[dots < 0] *= -1
                    cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    radii[valid] = r
                
                if np.any(~valid):
                    a_deg = a[~valid]
                    b_deg = b[~valid]
                    c_deg = c[~valid]
                    d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                    d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                    d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                    radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                
                max_idx = np.argmax(radii)
                rad = radii[max_idx]
                face = faces[max_idx]
            except Exception:
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            ab = b - a
            ac = c - a
            c_dir = np.cross(ab, ac)
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    direction = p - 0.1 * mid
                    direction /= np.linalg.norm(direction)
                    pts[idx] = direction
                continue
            
            c_dir /= c_norm
            # Antipode of circumcenter
            antipode = -c_dir
            
            # Try multiple step sizes toward antipode
            improved = False
            for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                new_a = a + step * antipode
                new_b = b + step * antipode
                new_c = c + step * antipode
                
                new_a /= np.linalg.norm(new_a)
                new_b /= np.linalg.norm(new_b)
                new_c /= np.linalg.norm(new_c)
                
                pts[i] = new_a
                pts[j] = new_b
                pts[k] = new_c
                
                new_rad = compute_covering_radius_hull(pts)
                if new_rad < rad:
                    improved = True
                    break
                else:
                    # Restore
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
            
            if not improved:
                # Try moving vertices individually toward antipode
                for idx, p in [(i, a), (j, b), (k, c)]:
                    for step in [0.01, 0.005, 0.002]:
                        new_p = p + step * antipode
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                        new_rad = compute_covering_radius_hull(pts)
                        if new_rad < rad:
                            improved = True
                            break
                        else:
                            pts[idx] = p
                    if improved:
                        break
            
            if not improved:
                # Try small random perturbations in tangent plane to escape local minima
                rng = np.random.RandomState(it)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    for _ in range(5):
                        # Random tangent direction
                        rand_vec = rng.randn(3)
                        tangent = rand_vec - np.dot(rand_vec, p) * p
                        t_norm = np.linalg.norm(tangent)
                        if t_norm < 1e-10:
                            continue
                        tangent /= t_norm
                        new_p = p + 0.005 * tangent
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                        new_rad = compute_covering_radius_hull(pts)
                        if new_rad < rad:
                            improved = True
                            break
                        else:
                            pts[idx] = p
                    if improved:
                        break
            
            if not improved:
                break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Initialization and L-BFGS-B optimization with smooth soft-max surrogate
    # We use the grid-based soft-max objective to guide the points initially.
    
    points = fibonacci_spiral(N).astype(np.float64)
    
    # Run L-BFGS-B
    x0 = points.ravel()
    beta = 50.0
    
    def objective_wrapper(x):
        val, _ = soft_max_objective(x, beta=beta)
        return val
    
    def gradient_wrapper(x):
        _, grad = soft_max_objective(x, beta=beta)
        return grad
        
    try:
        result = minimize(
            objective_wrapper,
            x0,
            jac=gradient_wrapper,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 200}
        )
        candidate = result.x.reshape(N, 3)
        norms = np.linalg.norm(candidate, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        candidate = candidate / norms
        rad = compute_covering_radius_hull(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()
    except Exception:
        pass

    if best_points is None:
        best_points = points.copy()
        best_rad = compute_covering_radius_hull(best_points)

    # Phase 2: Geometric local search (Targeted Refinement)
    # This is the most critical phase for achieving high precision.
    if time.time() < deadline - 10.0:
        refined = targeted_refinement(best_points.copy(), max_iters=500)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    # Phase 3: Final Polish with very small steps
    if time.time() < deadline - 5.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    return best_points

# EVOLVE-BLOCK-END
