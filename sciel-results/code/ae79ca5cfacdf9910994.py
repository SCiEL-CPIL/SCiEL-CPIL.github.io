"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

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
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
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

    # Precompute dense spherical grid for smooth surrogate
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
        Input: flat array of 3N coordinates.
        Returns: (objective_value, gradient)
        """
        pts = flat_pts.reshape(N, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Compute angular distances from each grid point to each configuration point
        # dot product matrix: (G, N)
        dots = grid_points @ pts.T  # shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Angular distance: arccos(dot)
        # For soft-max, we use: (1/beta) * log(sum(exp(-beta * dist)))
        # But we want to maximize min distance, so minimize max distance
        # min_dist for each grid point = min over N points of arccos(dot)
        # We approximate with soft-min: -(1/beta) * log(sum(exp(-beta * dist)))
        
        # Actually, let's use: covering_radius ≈ (1/beta) * log(sum(exp(beta * dist)))
        # where dist[i,j] = angular distance from grid point i to config point j
        # min over j for each i, then max over i
        
        # Simpler: use the fact that maximizing the minimum distance is equivalent
        # to minimizing the maximum distance. We can use:
        # f = (1/beta) * log(sum_i exp(beta * min_j d_ij))
        # But min_j is also non-smooth. Use soft-min for inner, soft-max for outer.
        
        # For efficiency, let's use a different approach:
        # The covering radius is the largest circumradius of hull facets.
        # As a smooth surrogate, we use:
        # f(beta, pts) = (1/beta) * log(sum over grid points i of exp(beta * min_j angular_dist(i, pts_j)))
        
        # Compute angular distances
        dists = np.arccos(dots)  # shape (G, N)
        
        # Soft-min over j (config points) for each grid point i
        # soft_min = -(1/beta) * log(sum_j exp(-beta * dists[i,j]))
        neg_beta_dists = -beta * dists
        # For numerical stability, subtract max
        max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
        log_sum_exp_inner = np.log(np.sum(np.exp(neg_beta_dists - max_neg), axis=1)) + max_neg.ravel()
        soft_min = -log_sum_exp_inner / beta  # shape (G,)
        
        # Soft-max over grid points i
        beta_soft_min = beta * soft_min
        max_outer = np.max(beta_soft_min)
        log_sum_exp_outer = np.log(np.sum(np.exp(beta_soft_min - max_outer))) + max_outer
        objective = log_sum_exp_outer / beta
        
        # Gradient computation
        # d(objective)/d(pts) requires chain rule through soft_min and dists
        # This is complex; use finite differences or analytical gradient
        
        # For L-BFGS-B, we can use numerical gradient or compute analytically
        # Let's compute the gradient analytically:
        # soft_min[i] = -(1/beta) * log(sum_j exp(-beta * d_ij))
        # d(soft_min[i])/d(pts_k) = (1/beta) * sum_j [exp(-beta*d_ij) / sum_j' exp(-beta*d_ij')] * (-beta) * d(d_ij)/d(pts_k)
        #                           = -sum_j [exp(-beta*d_ij) / sum_j' exp(-beta*d_ij')] * d(d_ij)/d(pts_k)
        # d(arccos(dot_ij))/d(pts_k) = -dot_ij / (sqrt(1-dot_ij^2)) * (grid_point_i / |grid_point_i|)
        #                            = -dot_ij / sqrt(1-dot_ij^2) * grid_point_i  (since |grid_point_i|=1)
        # But only for j=k, otherwise 0
        
        # So d(soft_min[i])/d(pts_k) = -w_ik * (-dot_ik / sqrt(1-dot_ik^2)) * grid_point_i
        # where w_ik = exp(-beta*d_ik) / sum_j' exp(-beta*d_ij')
        #            = exp(neg_beta_dists[i,k]) / sum_j' exp(neg_beta_dists[i,j'])
        
        # Then d(objective)/d(pts_k) = sum_i [w_outer_i * d(soft_min[i])/d(pts_k)]
        # where w_outer_i = exp(beta*soft_min[i]) / sum_i' exp(beta*soft_min[i'])
        
        # Weights for outer soft-max
        outer_weights = np.exp(beta_soft_min - max_outer)
        outer_weights /= np.sum(outer_weights)  # shape (G,)
        
        # Weights for inner soft-min
        inner_weights = np.exp(neg_beta_dists - max_neg)  # shape (G, N)
        inner_weights /= np.sum(inner_weights, axis=1, keepdims=True)  # shape (G, N)
        
        # d(soft_min[i])/d(pts_k) = inner_weights[i,k] * dot_ik / sqrt(1-dot_ik^2) * grid_point_i
        # Note: the two negatives cancel
        
        # Avoid division by zero
        one_minus_dot_sq = 1.0 - dots * dots
        one_minus_dot_sq = np.maximum(one_minus_dot_sq, 1e-12)
        sqrt_oms = np.sqrt(one_minus_dot_sq)
        
        # grad_soft_min[i,k] = inner_weights[i,k] * dots[i,k] / sqrt_oms[i,k] * grid_point_i
        # Shape: (G, N, 3)
        coeff = inner_weights * dots / sqrt_oms  # shape (G, N)
        
        # d(objective)/d(pts_k) = sum_i outer_weights[i] * coeff[i,k] * grid_point_i
        # = (coeff.T @ (outer_weights * grid_points)) ... no
        # = sum_i outer_weights[i] * coeff[i,k] * grid_points[i,:]
        
        # grad_pts shape (N, 3)
        # grad_pts[k,:] = sum_i outer_weights[i] * coeff[i,k] * grid_points[i,:]
        # = (coeff * outer_weights[:, np.newaxis]).T @ grid_points
        
        weighted_coeff = coeff * outer_weights[:, np.newaxis]  # shape (G, N)
        grad_pts = weighted_coeff.T @ grid_points  # shape (N, 3)
        
        # Now convert from pts gradient to flat_pts gradient
        # pts = flat_pts / norms, so d(pts)/d(flat_pts) involves the projection
        # For simplicity, use the gradient as-is (approximation) or compute properly
        # Proper: if pts = x / ||x||, then d(pts)/d(x) = (I - pts*pts^T) / ||x||
        # So grad_flat = (I - pts*pts^T) @ grad_pts / norms
        
        # Actually, for L-BFGS-B on the sphere, we can just use the tangent space gradient
        # Project grad_pts to tangent space: grad_tangent = grad_pts - (grad_pts . pts) * pts
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
        
        # Convert to flat gradient
        grad_flat = grad_tangent.ravel()
        
        return objective, grad_flat

    def repulsion_force(points, exponent=1.5):
        """Compute repulsion forces between all point pairs."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        return forces

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def targeted_refinement(points, max_iters=200):
        """Refine by targeting the worst facet: move its 3 vertices to reduce circumradius."""
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
                
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norms = np.linalg.norm(c_dir, axis=1)
                
                radii = np.zeros(len(faces))
                valid = norms > 1e-10
                
                if np.any(valid):
                    c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                    cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
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
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
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
            step = 0.005
            for idx, p in [(i, a), (j, b), (k, c)]:
                tangent = c_dir - np.dot(c_dir, p) * p
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent /= t_norm
                    new_p = p + step * tangent
                    new_p /= np.linalg.norm(new_p)
                    pts[idx] = new_p
            
            # Check improvement
            new_rad = compute_covering_radius_hull(pts)
            if new_rad >= rad:
                pts[i] = a
                pts[j] = b
                pts[k] = c
                step *= 0.5
                if step < 1e-6:
                    break
                for idx, p in [(i, a), (j, b), (k, c)]:
                    tangent = c_dir - np.dot(c_dir, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-10:
                        tangent /= t_norm
                        new_p = p + step * tangent
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                new_rad = compute_covering_radius_hull(pts)
                if new_rad >= rad:
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
                    break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion-based gradient descent (fast initialization)
    n_restarts = 4
    exponents = [1.0, 1.5, 2.0, 2.5]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 150, exp)
        rad = compute_covering_radius_hull(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: L-BFGS-B optimization with smooth soft-max surrogate
    if best_points is not None and time.time() < deadline - 30.0:
        for beta in [30.0, 50.0, 80.0]:
            if time.time() > deadline - 15.0:
                break
            
            # Use the current best as starting point
            x0 = best_points.ravel()
            
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
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 100}
                )
                
                if result.success or result.fun < objective_wrapper(x0):
                    candidate = result.x.reshape(N, 3)
                    # Normalize
                    norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    candidate = candidate / norms
                    rad = compute_covering_radius_hull(candidate)
                    if rad < best_rad:
                        best_rad = rad
                        best_points = candidate.copy()
            except Exception:
                pass

    # Phase 3: Targeted refinement on the worst facet
    if best_points is not None and time.time() < deadline - 10.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 4: Final L-BFGS-B polish with higher beta
    if best_points is not None and time.time() < deadline - 8.0:
        beta = 100.0
        x0 = best_points.ravel()
        
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
                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 60}
            )
            
            if result.success or result.fun < objective_wrapper(x0):
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

    # Final targeted refinement
    if best_points is not None and time.time() < deadline - 5.0:
        refined = targeted_refinement(best_points.copy(), max_iters=100)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
