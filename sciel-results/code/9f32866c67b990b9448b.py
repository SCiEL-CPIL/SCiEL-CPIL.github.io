"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

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
            
            # For each face, compute the circumcenter (on the sphere)
            # The circumradius is the angular distance from the face center to any vertex
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
            # Fallback: dense grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 100)
            lon = np.linspace(0, 2*np.pi, 200, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
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

    def make_fib_grid(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    grid_pts_coarse = make_fib_grid(1500)
    grid_pts_fine = make_fib_grid(5000)

    def grid_objective_and_grad(x_flat, grid_pts, temperature=50.0):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        Uses log-sum-exp for numerical stability. Fully vectorized.
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
        
        # Gradient computation - fully vectorized
        argmin_k = np.argmin(ang_dists, axis=1)  # M
        weights = exp_shifted / sum_exp  # M
        
        # For each grid point j, nearest config point is k=argmin_k[j]
        # dot_val[j] = grid_pts[j] · pts_norm[k]
        # We need: grad_pts_norm[k] += weights[j] * (-grid_pts[j] / sqrt(1 - dot_val[j]^2))
        
        # Gather the nearest points
        nearest_pts = pts_norm[argmin_k]  # M x 3
        dot_vals = np.sum(grid_pts * nearest_pts, axis=1)  # M
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        denom = np.sqrt(np.maximum(1.0 - dot_vals * dot_vals, 1e-10))  # M
        
        # Weighted gradient contribution per grid point
        weighted_g = (weights / denom)[:, np.newaxis] * grid_pts  # M x 3
        # We want to move p_k in direction -g/denom (to increase arccos)
        # So the gradient contribution is -weighted_g accumulated per k
        neg_weighted_g = -weighted_g  # M x 3
        
        # Accumulate per point index
        grad_pts_norm = np.zeros((N, 3))
        np.add.at(grad_pts_norm, argmin_k, neg_weighted_g)
        
        # Project gradient to tangent space and chain rule
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            grad_tangent = grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p
            grad[i*3:(i+1)*3] = grad_tangent / n
        
        return float(log_sum_exp), grad

    def optimize_with_lbfgsb(points, max_time=30.0, grid_pts=None, temperature=50.0):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        call_count = [0]
        
        def objective(x):
            call_count[0] += 1
            val, _ = grid_objective_and_grad(x, grid_pts, temperature)
            return val
        
        def gradient(x):
            call_count[0] += 1
            _, g = grid_objective_and_grad(x, grid_pts, temperature)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        # Estimate how many iterations we can afford
        # Each call takes roughly M*N*some_constant time
        M = len(grid_pts)
        est_time_per_call = M * N * 3e-7  # rough estimate
        max_iter = max(50, min(500, int(max_time / max(est_time_per_call, 0.01))))
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def facet_local_search(points, max_rounds=10, max_time=None):
        """
        Local search that directly targets the covering radius by moving
        points to reduce the largest facet circumradius.
        """
        if max_time is None:
            max_time = 10.0
        start_t = time.time()
        points = points.copy()
        
        for _round in range(max_rounds):
            if time.time() - start_t > max_time:
                break
            
            # Compute current covering radius and identify worst facet
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            faces = hull.simplices
            a = points[faces[:, 0]]
            b = points[faces[:, 1]]
            c = points[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms_c = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms_c > 1e-10
            if np.any(valid):
                c_dir_v = c_dir[valid] / norms_c[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_v * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            if np.any(~valid):
                d_ab = np.arccos(np.clip(np.sum(a[~valid] * b[~valid], axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b[~valid] * c[~valid], axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c[~valid] * a[~valid], axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            # Find the worst facet (largest radius)
            worst_idx = np.argmax(radii)
            worst_rad = radii[worst_idx]
            worst_face = faces[worst_idx]
            v0, v1, v2 = worst_face
            
            # The circumcenter direction
            a_w = points[v0]
            b_w = points[v1]
            c_w = points[v2]
            ab_w = np.cross(a_w, b_w)
            bc_w = np.cross(b_w, c_w)
            ca_w = np.cross(c_w, a_w)
            c_center_dir = ab_w + bc_w + ca_w
            c_center_norm = np.linalg.norm(c_center_dir)
            
            if c_center_norm < 1e-10:
                # Degenerate: move the point farthest from centroid
                centroid = (a_w + b_w + c_w) / 3.0
                centroid /= np.linalg.norm(centroid)
                dists = [np.arccos(np.clip(np.dot(v, centroid), -1, 1)) for v in [a_w, b_w, c_w]]
                move_idx = [v0, v1, v2][np.argmax(dists)]
                # Move toward the circumcenter direction
                target = -points[move_idx]  # move away from itself, toward antipode... 
                # Better: move toward the face center
                target = centroid
                target -= np.dot(target, points[move_idx]) * points[move_idx]
                target /= np.linalg.norm(target)
                for step in [0.02, 0.01, 0.005]:
                    new_p = points[move_idx] + step * target
                    new_p /= np.linalg.norm(new_p)
                    old = points[move_idx].copy()
                    points[move_idx] = new_p
                    try:
                        hull2 = ConvexHull(points)
                        faces2 = hull2.simplices
                        a2 = points[faces2[:, 0]]
                        b2 = points[faces2[:, 1]]
                        c2 = points[faces2[:, 2]]
                        ab2 = np.cross(a2, b2)
                        bc2 = np.cross(b2, c2)
                        ca2 = np.cross(c2, a2)
                        cd2 = ab2 + bc2 + ca2
                        nc2 = np.linalg.norm(cd2, axis=1)
                        rad2 = np.zeros(len(faces2))
                        v2 = nc2 > 1e-10
                        if np.any(v2):
                            cdv = cd2[v2] / nc2[v2, np.newaxis]
                            cr = np.clip(np.sum(cdv * a2[v2], axis=1), -1, 1)
                            rr = np.arccos(cr)
                            rr[rr > np.pi/2] = np.pi - rr[rr > np.pi/2]
                            rad2[v2] = rr
                        if np.any(~v2):
                            dab = np.arccos(np.clip(np.sum(a2[~v2]*b2[~v2], axis=1), -1, 1))
                            dbc = np.arccos(np.clip(np.sum(b2[~v2]*c2[~v2], axis=1), -1, 1))
                            dca = np.arccos(np.clip(np.sum(c2[~v2]*a2[~v2], axis=1), -1, 1))
                            rad2[~v2] = np.maximum(np.maximum(dab, dbc), dca)
                        if np.max(rad2) < worst_rad:
                            break
                    except Exception:
                        points[move_idx] = old
                        break
                else:
                    points[move_idx] = old
                continue
            
            c_center = c_center_dir / c_center_norm
            
            # Try moving each vertex of the worst facet toward the circumcenter
            improved_this_round = False
            for vi, v_idx in enumerate([v0, v1, v2]):
                p = points[v_idx]
                # Direction: toward circumcenter, projected to tangent
                target = c_center
                target -= np.dot(target, p) * p
                target_norm = np.linalg.norm(target)
                if target_norm < 1e-10:
                    continue
                target /= target_norm
                
                for step in [0.03, 0.015, 0.008, 0.004]:
                    new_p = p + step * target
                    new_p /= np.linalg.norm(new_p)
                    old = points[v_idx].copy()
                    points[v_idx] = new_p
                    try:
                        hull2 = ConvexHull(points)
                        faces2 = hull2.simplices
                        a2 = points[faces2[:, 0]]
                        b2 = points[faces2[:, 1]]
                        c2 = points[faces2[:, 2]]
                        ab2 = np.cross(a2, b2)
                        bc2 = np.cross(b2, c2)
                        ca2 = np.cross(c2, a2)
                        cd2 = ab2 + bc2 + ca2
                        nc2 = np.linalg.norm(cd2, axis=1)
                        rad2 = np.zeros(len(faces2))
                        v2 = nc2 > 1e-10
                        if np.any(v2):
                            cdv = cd2[v2] / nc2[v2, np.newaxis]
                            cr = np.clip(np.sum(cdv * a2[v2], axis=1), -1, 1)
                            rr = np.arccos(cr)
                            rr[rr > np.pi/2] = np.pi - rr[rr > np.pi/2]
                            rad2[v2] = rr
                        if np.any(~v2):
                            dab = np.arccos(np.clip(np.sum(a2[~v2]*b2[~v2], axis=1), -1, 1))
                            dbc = np.arccos(np.clip(np.sum(b2[~v2]*c2[~v2], axis=1), -1, 1))
                            dca = np.arccos(np.clip(np.sum(c2[~v2]*a2[~v2], axis=1), -1, 1))
                            rad2[~v2] = np.maximum(np.maximum(dab, dbc), dca)
                        if np.max(rad2) < worst_rad - 1e-12:
                            improved_this_round = True
                            break
                    except Exception:
                        points[v_idx] = old
                        break
                else:
                    points[v_idx] = old
                
                if improved_this_round:
                    break
            
            if not improved_this_round:
                break
        
        return points

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 45.0:
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
        elif restart == 4:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(400 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 5:
            rng = np.random.RandomState(300 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 6:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(600 + restart)
            points = points + 0.2 * rng.randn(N, 3)
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
        time_left = deadline - time.time() - 20.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, max_time=min(time_left, 25.0), grid_pts=grid_pts_coarse, temperature=50.0)
        
        # Facet-based local search
        time_left = deadline - time.time() - 15.0
        if time_left > 5.0:
            points = facet_local_search(points, max_rounds=8, max_time=min(time_left, 10.0))
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with facet-based search and fine grid L-BFGS-B
    if best_points is not None and time.time() < deadline - 25.0:
        for trial in range(6):
            if time.time() > deadline - 15.0:
                break
            rng = np.random.RandomState(700 + trial)
            noise_scale = 0.02 + 0.01 * trial
            candidate = best_points + noise_scale * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 12.0
            if time_left > 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, max_time=min(time_left, 15.0), grid_pts=grid_pts_fine, temperature=100.0)
            
            # Facet-based local search
            time_left = deadline - time.time() - 8.0
            if time_left > 3.0:
                candidate = facet_local_search(candidate, max_rounds=10, max_time=min(time_left, 8.0))
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Facet-based local search with multiple passes
    if best_points is not None and time.time() < deadline - 12.0:
        points = best_points.copy()
        time_left = deadline - time.time() - 8.0
        points = facet_local_search(points, max_rounds=15, max_time=min(time_left, 12.0))
        rad = compute_covering_radius_exact(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 4: Final polish with L-BFGS-B using fine grid and higher temperature
    if best_points is not None and time.time() < deadline - 6.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 4.0
        if time_left > 4.0:
            final, _ = optimize_with_lbfgsb(final, max_time=min(time_left, 10.0), grid_pts=grid_pts_fine, temperature=150.0)
            # Final facet search
            time_left2 = deadline - time.time() - 2.0
            if time_left2 > 2.0:
                final = facet_local_search(final, max_rounds=10, max_time=min(time_left2, 6.0))
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
