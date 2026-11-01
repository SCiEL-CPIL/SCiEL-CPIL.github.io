"""Program for covering the sphere with 130 equal caps using hybrid grid and facet-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute grids for the smooth objective
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(3000)
GRID_FINE = _make_fib_grid(10000)

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

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
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
                # Circumradius on sphere is the angle from circumcenter to vertex.
                # If the triangle contains the center, the circumcenter is on the opposite side?
                # No, for a spherical triangle, the circumcenter is unique.
                # The formula arccos(u . a) gives the angle.
                # If the triangle is large (contains pole), the circumcenter might be "far".
                # However, for covering radius, we care about the empty cap.
                # The empty cap is centered at the circumcenter of the facet.
                # The radius is the angular distance to the nearest vertex.
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
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def repulsion_energy_grad(pts, exponent=2.0):
        """Compute repulsion energy and gradient for uniform distribution."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def grid_objective_and_grad(x_flat, grid_pts, temperature=50.0):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N

        # For each grid point, find the min distance to any configuration point
        min_dists = np.min(ang_dists, axis=1)  # M
        
        # Soft-max of min_dists to get a smooth approximation of the covering radius
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(shifted * temperature)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = np.log(sum_exp) / temperature + max_d
        
        # Gradient computation
        argmin_k = np.argmin(ang_dists, axis=1)  # M
        weights = exp_shifted / sum_exp  # M
        
        grad_pts_norm = np.zeros((N, 3))
        
        # Vectorized gradient accumulation
        # For each grid point j, the gradient contribution to point k=argmin_k[j] is:
        # w_j * d/dp_k (arccos(g_j . p_k))
        # = w_j * (-g_j / sqrt(1 - (g_j . p_k)^2))
        
        # We need to accumulate these into grad_pts_norm[k]
        # Using np.add.at for efficient accumulation
        g = grid_pts[argmin_k] # M x 3
        p = pts_norm[argmin_k] # M x 3
        dot_val = np.sum(g * p, axis=1) # M
        dot_val = np.clip(dot_val, -1.0, 1.0)
        denom = np.sqrt(np.maximum(1.0 - dot_val * dot_val, 1e-10)) # M
        grad_vec = -g / denom[:, np.newaxis] # M x 3
        grad_vec *= weights[:, np.newaxis] # M x 3
        
        np.add.at(grad_pts_norm, argmin_k, grad_vec)
        
        # Project gradient to tangent space and chain rule
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # Project to tangent space: (I - p p^T)
            grad_tangent = grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p
            # Chain rule: d(pts_norm[i])/d(pts[i]) = (I - p p^T)/n
            grad[i*3:(i+1)*3] = grad_tangent / n
        
        return float(log_sum_exp), grad

    def optimize_with_lbfgsb(points, grid_pts, temperature=50.0, max_iter=200):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = grid_objective_and_grad(x, grid_pts, temperature)
            return val
        
        def gradient(x):
            _, g = grid_objective_and_grad(x, grid_pts, temperature)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 4, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def circumradius(a, b, c):
        """Calculate circumradius of triangle on sphere."""
        s = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-10:
            return np.pi / 2 # Degenerate, assume large
        u = s / s_norm
        cos_r = np.clip(np.dot(u, a), -1.0, 1.0)
        return np.arccos(cos_r)

    def facet_local_search(pts, max_time=30.0):
        """
        Local search using numerical gradients of the circumradius of the worst facets.
        """
        pts = pts.copy()
        current_rad = compute_covering_radius_exact(pts)
        alpha = 0.05
        min_alpha = 1e-5
        no_improve_count = 0
        
        while time.time() < max_time and no_improve_count < 20:
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                # Calculate radii for all facets
                radii = np.array([circumradius(a[i], b[i], c[i]) for i in range(len(faces))])
                max_rad = np.max(radii)
                
                # Identify active facets (those close to the max)
                active_indices = np.where(radii > max_rad - 1e-4)[0]
                
                if len(active_indices) == 0:
                    break
                    
                # Compute gradients for active facets
                grad_acc = np.zeros_like(pts)
                
                for idx in active_indices:
                    i, j, k = faces[idx]
                    a_i = pts[i]
                    b_j = pts[j]
                    c_k = pts[k]
                    
                    # Numerical gradient for point i
                    h = 1e-4
                    # Tangent basis for a_i
                    t1 = np.cross(a_i, np.array([1.0, 0.0, 0.0]))
                    if np.linalg.norm(t1) < 1e-6:
                        t1 = np.cross(a_i, np.array([0.0, 1.0, 0.0]))
                    t1 /= np.linalg.norm(t1)
                    t2 = np.cross(a_i, t1)
                    t2 /= np.linalg.norm(t2)
                    
                    a_p1 = a_i + h * t1
                    a_p1 /= np.linalg.norm(a_p1)
                    a_m1 = a_i - h * t1
                    a_m1 /= np.linalg.norm(a_m1)
                    a_p2 = a_i + h * t2
                    a_p2 /= np.linalg.norm(a_p2)
                    a_m2 = a_i - h * t2
                    a_m2 /= np.linalg.norm(a_m2)
                    
                    r_p1 = circumradius(a_p1, b_j, c_k)
                    r_m1 = circumradius(a_m1, b_j, c_k)
                    r_p2 = circumradius(a_p2, b_j, c_k)
                    r_m2 = circumradius(a_m2, b_j, c_k)
                    
                    grad_i = ((r_p1 - r_m1) / (2*h)) * t1 + ((r_p2 - r_m2) / (2*h)) * t2
                    grad_acc[i] += grad_i
                    
                    # Similar for j and k (simplified: just use same logic)
                    # For brevity and speed, we might skip full gradient for all 3 in tight loop if time is short,
                    # but let's do it for correctness.
                    
                    # Point j
                    t1 = np.cross(b_j, np.array([1.0, 0.0, 0.0]))
                    if np.linalg.norm(t1) < 1e-6:
                        t1 = np.cross(b_j, np.array([0.0, 1.0, 0.0]))
                    t1 /= np.linalg.norm(t1)
                    t2 = np.cross(b_j, t1)
                    t2 /= np.linalg.norm(t2)
                    
                    b_p1 = b_j + h * t1
                    b_p1 /= np.linalg.norm(b_p1)
                    b_m1 = b_j - h * t1
                    b_m1 /= np.linalg.norm(b_m1)
                    b_p2 = b_j + h * t2
                    b_p2 /= np.linalg.norm(b_p2)
                    b_m2 = b_j - h * t2
                    b_m2 /= np.linalg.norm(b_m2)
                    
                    r_p1 = circumradius(a_i, b_p1, c_k)
                    r_m1 = circumradius(a_i, b_m1, c_k)
                    r_p2 = circumradius(a_i, b_p2, c_k)
                    r_m2 = circumradius(a_i, b_m2, c_k)
                    
                    grad_j = ((r_p1 - r_m1) / (2*h)) * t1 + ((r_p2 - r_m2) / (2*h)) * t2
                    grad_acc[j] += grad_j
                    
                    # Point k
                    t1 = np.cross(c_k, np.array([1.0, 0.0, 0.0]))
                    if np.linalg.norm(t1) < 1e-6:
                        t1 = np.cross(c_k, np.array([0.0, 1.0, 0.0]))
                    t1 /= np.linalg.norm(t1)
                    t2 = np.cross(c_k, t1)
                    t2 /= np.linalg.norm(t2)
                    
                    c_p1 = c_k + h * t1
                    c_p1 /= np.linalg.norm(c_p1)
                    c_m1 = c_k - h * t1
                    c_m1 /= np.linalg.norm(c_m1)
                    c_p2 = c_k + h * t2
                    c_p2 /= np.linalg.norm(c_p2)
                    c_m2 = c_k - h * t2
                    c_m2 /= np.linalg.norm(c_m2)
                    
                    r_p1 = circumradius(a_i, b_j, c_p1)
                    r_m1 = circumradius(a_i, b_j, c_m1)
                    r_p2 = circumradius(a_i, b_j, c_p2)
                    r_m2 = circumradius(a_i, b_j, c_m2)
                    
                    grad_k = ((r_p1 - r_m1) / (2*h)) * t1 + ((r_p2 - r_m2) / (2*h)) * t2
                    grad_acc[k] += grad_k

                # Update points
                pts_new = pts - alpha * grad_acc
                pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
                
                new_rad = compute_covering_radius_exact(pts_new)
                
                if new_rad < current_rad - 1e-10:
                    pts = pts_new
                    current_rad = new_rad
                    no_improve_count = 0
                    # Slightly increase alpha if improving
                    if alpha < 0.1:
                        alpha *= 1.05
                else:
                    no_improve_count += 1
                    alpha *= 0.8
                    if alpha < min_alpha:
                        break
                        
            except Exception:
                break
                
        return pts, current_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 2.0, 0.1, 0.99, 100)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 30.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, temperature=50.0, max_iter=100)
            
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 30.0:
        for trial in range(3):
            if time.time() > deadline - 20.0:
                break
            rng = np.random.RandomState(700 + trial)
            candidate = best_points + 0.02 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 15.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, temperature=100.0, max_iter=50)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Facet-based local search
    if best_points is not None and time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 3.0
        if time_left > 5.0:
            best_points, best_rad = facet_local_search(best_points, max_time=time_left)

    # Phase 4: Final check
    if best_points is not None:
        final_rad = compute_covering_radius_exact(best_points)
        if final_rad < best_rad:
            best_rad = final_rad

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
