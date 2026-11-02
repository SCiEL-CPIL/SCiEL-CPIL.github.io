"""Program for covering the sphere with 130 equal caps using repulsion and facet-local search."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

def _project_tangent(v, p):
    """Project vector v onto tangent plane at point p (unit vector)."""
    return v - np.dot(v, p) * p

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety

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
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def repulsion_energy_grad(pts, exponent=2.0):
        """Compute repulsion energy and gradient for uniform distribution.
        Uses Riesz s-energy with high s to approximate covering radius minimization."""
        s = exponent
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        # E = sum_{i<j} d_{ij}^{-s} = sum_{i<j} dist_sq_{ij}^{-s/2}
        w = dist_sq ** (-s / 2.0)
        np.fill_diagonal(w, 0.0)
        
        # dE/ddist_sq = -s/2 * dist_sq^{-(s/2+1)}
        # d(dist_sq)/d(pts) involves the gradient
        w_grad_coeff = (s / 2.0) * (dist_sq ** (-(s / 2.0) - 1))
        np.fill_diagonal(w_grad_coeff, 0.0)
        
        # grad = w_grad_coeff @ pts (each point contributes to all others)
        grad = w_grad_coeff @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy with projection to sphere."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            
            # Project gradient to tangent space
            grad_tangent = np.array([_project_tangent(grad[i], pts[i]) for i in range(N)])
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm > 1e-10:
                grad_tangent = grad_tangent / grad_norm
            
            pts = pts + alpha * grad_tangent
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def grid_objective_and_grad(x_flat, grid_pts):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        Uses a vectorized, differentiable approximation.
        x_flat: flattened (N,3) array of points.
        grid_pts: precomputed grid of sphere points.
        Returns (value, gradient).
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

        # 1. Soft-min over points for each grid point
        # s_j = -1/T_min * log( sum_k exp(-T_min * ang_dists[j, k]) )
        T_min = 80.0
        # Use log-sum-exp for numerical stability
        neg_T_min_d = -T_min * ang_dists
        max_neg_T_min_d = np.max(neg_T_min_d, axis=1, keepdims=True)
        exp_neg_T_min_d = np.exp(neg_T_min_d - max_neg_T_min_d)
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1)
        log_sum_exp_neg = max_neg_T_min_d[:, 0] + np.log(sum_exp_neg)
        soft_min_dists = log_sum_exp_neg / T_min  # Note: -1/T_min * log(sum) = (1/T_min) * log_sum_exp_neg
        
        # Weights for soft-min: w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis] # M x N

        # 2. Soft-max over grid points
        # val = max_d + 1/T_max * log(sum_j exp(T_max * (s_j - max_d)))
        T_max = 80.0
        max_s = np.max(soft_min_dists)
        T_max_s = T_max * (soft_min_dists - max_s)
        exp_T_max_s = np.exp(T_max_s)
        sum_exp_pos = np.sum(exp_T_max_s)
        val = max_s + 1.0 / T_max * np.log(sum_exp_pos)
        
        # Weights for soft-max: v_j = exp(T_max * (s_j - max_s)) / sum_exp_pos
        v_j = exp_T_max_s / sum_exp_pos # M

        # Gradient of val w.r.t pts_norm[k]
        # grad_pts_norm[k] = sum_j v_j * w_jk * (-grid_pts[j] / sqrt(1 - dot[j,k]^2))
        
        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10)) # M x N
        factor = (v_j[:, np.newaxis] * w_jk) / denom # M x N
        grad_pts_norm = -(factor.T @ grid_pts) # N x 3
        
        # Project to tangent space and apply chain rule for normalization
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # Project to tangent plane
            grad_tangent = _project_tangent(grad_pts_norm[i], p)
            # Chain rule: d(pts_norm[i])/d(pts[i]) = (I - p p^T)/n
            grad_i = grad_tangent / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = grid_objective_and_grad(x, grid_pts)
            return val
        
        def gradient(x):
            _, g = grid_objective_and_grad(x, grid_pts)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def facet_local_search(points, max_rounds=5):
        """
        Local search by perturbing points that belong to the largest facets.
        This targets the covering radius directly.
        """
        best_points = points.copy()
        best_rad = compute_covering_radius_exact(best_points)
        
        for round_idx in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            
            try:
                hull = ConvexHull(best_points)
                faces = hull.simplices
                a = best_points[faces[:, 0]]
                b = best_points[faces[:, 1]]
                c = best_points[faces[:, 2]]
                
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
                
                # Sort facets by radius, largest first
                sorted_indices = np.argsort(radii)[::-1]
                
                # Take top K facets
                K = 10
                top_indices = sorted_indices[:K]
                
                improved = False
                for idx in top_indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    face = faces[idx]
                    # Get unique points in this facet
                    unique_pts = np.unique(face)
                    
                    # For each point in the facet, try to move it to reduce the facet radius
                    for p_idx in unique_pts:
                        old_p = best_points[p_idx].copy()
                        
                        # Compute the normal of the facet (pointing outward)
                        # The circumcenter direction is c_dir / norm
                        if valid[idx]:
                            normal = c_dir[idx] / norms[idx]
                        else:
                            # Fallback normal
                            normal = np.cross(best_points[face[0]], best_points[face[1]])
                            normal /= np.linalg.norm(normal)
                        
                        # Let's try a small random perturbation and check if radius decreases.
                        rng = np.random.RandomState(p_idx + round_idx * 1000 + int(idx))
                        for trial in range(15):
                            direction = rng.randn(3)
                            # Project to tangent plane
                            direction = _project_tangent(direction, old_p)
                            dir_norm = np.linalg.norm(direction)
                            if dir_norm < 1e-10:
                                continue
                            direction /= dir_norm
                            
                            # Try small steps
                            for step in [0.01, 0.005, 0.001, 0.002]:
                                new_p = old_p + step * direction
                                new_p /= np.linalg.norm(new_p)
                                best_points[p_idx] = new_p
                                
                                new_rad = compute_covering_radius_exact(best_points)
                                if new_rad < best_rad - 1e-12:
                                    best_rad = new_rad
                                    improved = True
                                    break
                                else:
                                    best_points[p_idx] = old_p
                            else:
                                continue
                            break
                        
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            except Exception:
                break
        
        return best_points, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
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
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        else:
            # Use a higher exponent for Riesz energy to better approximate covering
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.05 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        # Use higher exponent for better covering approximation
        exponent = 10.0 if restart >= 3 else 2.0
        points = gradient_descent_repulsion(points, exponent, 0.05, 0.98, 150)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 15.0
        if time_left > 15.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=300)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Facet Local Search
    if best_points is not None and time.time() < deadline - 40.0:
        best_points, best_rad = facet_local_search(best_points, max_rounds=15)

    # Phase 3: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 30.0:
        for trial in range(4):
            if time.time() > deadline - 15.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.03 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 10.0
            if time_left > 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=150)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
                
                # Run facet local search again on the new best
                if time.time() < deadline - 15.0:
                    best_points, best_rad = facet_local_search(best_points, max_rounds=8)

    # Phase 4: Final polish with L-BFGS-B using fine grid
    if best_points is not None and time.time() < deadline - 10.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 5.0
        if time_left > 5.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=100)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
