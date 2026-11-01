"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute grids for fast evaluation
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

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

    def fast_covering_radius(pts, grid=None):
        """Fast grid-based covering radius estimate."""
        if grid is None:
            grid = GRID_COARSE
        dots = grid @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return float(np.max(min_dists))

    def soft_max_objective(flat_pts, beta=50.0, grid=None):
        """
        Smooth surrogate for covering radius using soft-max.
        Input: flat array of 3N coordinates.
        Returns: (objective_value, gradient)
        """
        if grid is None:
            grid = GRID_COARSE
            
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        dots = grid @ pts.T  # shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # shape (G, N)
        
        # Soft-min over j (config points) for each grid point i
        neg_beta_dists = -beta * dists
        max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
        log_sum_exp_inner = np.log(np.sum(np.exp(neg_beta_dists - max_neg), axis=1)) + max_neg.ravel()
        soft_min = -log_sum_exp_inner / beta  # shape (G,)
        
        # Soft-max over grid points i
        beta_soft_min = beta * soft_min
        max_outer = np.max(beta_soft_min)
        log_sum_exp_outer = np.log(np.sum(np.exp(beta_soft_min - max_outer))) + max_outer
        objective = log_sum_exp_outer / beta
        
        # Gradient computation
        outer_weights = np.exp(beta_soft_min - max_outer)
        outer_weights /= np.sum(outer_weights)  # shape (G,)
        
        inner_weights = np.exp(neg_beta_dists - max_neg)  # shape (G, N)
        inner_weights /= np.sum(inner_weights, axis=1, keepdims=True)  # shape (G, N)
        
        one_minus_dot_sq = 1.0 - dots * dots
        one_minus_dot_sq = np.maximum(one_minus_dot_sq, 1e-12)
        sqrt_oms = np.sqrt(one_minus_dot_sq)
        
        coeff = inner_weights * dots / sqrt_oms  # shape (G, N)
        
        weighted_coeff = coeff * outer_weights[:, np.newaxis]  # shape (G, N)
        grad_pts = weighted_coeff.T @ grid  # shape (N, 3)
        
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
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
                break
        
        return pts

    def lbfgs_optimize(points, beta, max_iter=50, ftol=1e-10, gtol=1e-8, grid=None):
        """Run L-BFGS-B optimization with the soft-max surrogate."""
        if grid is None:
            grid = GRID_COARSE
            
        x0 = points.ravel()
        
        def objective_wrapper(x):
            val, _ = soft_max_objective(x, beta=beta, grid=grid)
            return val
        
        def gradient_wrapper(x):
            _, grad = soft_max_objective(x, beta=beta, grid=grid)
            return grad
        
        try:
            result = minimize(
                objective_wrapper,
                x0,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxfun': max_iter * 2}
            )
            
            if result.success or result.fun < objective_wrapper(x0):
                candidate = result.x.reshape(N, 3)
                norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                candidate = candidate / norms
                return candidate
        except Exception:
            pass
        
        return points

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
        
        # Use fast grid-based evaluation for ranking
        rad_fast = fast_covering_radius(points, GRID_FINE)
        
        if best_points is None or rad_fast < best_rad:
            # Verify with exact hull computation
            rad_exact = compute_covering_radius_hull(points)
            if rad_exact < best_rad:
                best_rad = rad_exact
                best_points = points.copy()

    # Phase 2: L-BFGS-B optimization with smooth soft-max surrogate
    if best_points is not None and time.time() < deadline - 35.0:
        for beta in [30.0, 50.0, 80.0]:
            if time.time() > deadline - 20.0:
                break
            
            # Use the current best as starting point
            candidate = lbfgs_optimize(best_points, beta, max_iter=50, ftol=1e-10, gtol=1e-8, grid=GRID_COARSE)
            rad = compute_covering_radius_hull(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Targeted refinement on the worst facet (multiple passes)
    if best_points is not None and time.time() < deadline - 15.0:
        for pass_num in range(3):
            if time.time() > deadline - 10.0:
                break
            refined = targeted_refinement(best_points.copy(), max_iters=150)
            rad = compute_covering_radius_hull(refined)
            if rad < best_rad:
                best_rad = rad
                best_points = refined.copy()
            else:
                break

    # Phase 4: Final L-BFGS-B polish with higher beta and finer grid
    if best_points is not None and time.time() < deadline - 12.0:
        beta = 100.0
        candidate = lbfgs_optimize(best_points, beta, max_iter=40, ftol=1e-12, gtol=1e-10, grid=GRID_FINE)
        rad = compute_covering_radius_hull(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()

    # Final targeted refinement
    if best_points is not None and time.time() < deadline - 6.0:
        refined = targeted_refinement(best_points.copy(), max_iters=150)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
