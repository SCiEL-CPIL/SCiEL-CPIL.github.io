"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
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
GRID_VERY_FINE = _make_fib_grid(15000)

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

    def repulsion_energy_grad(pts, exponent=4.0):
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

    def grid_objective_and_grad(x_flat, grid_pts, T=80.0):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        Uses hard min over points (differentiable a.e.) and soft-max over grid points.
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

        # Hard min over points for each grid point
        min_dists = np.min(ang_dists, axis=1)  # M
        min_indices = np.argmin(ang_dists, axis=1)  # M

        # Soft-max over grid points
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(T * shifted)
        sum_exp = np.sum(exp_shifted)
        val = max_d + (1.0 / T) * np.log(sum_exp)
        
        # Weights for soft-max
        weights = exp_shifted / sum_exp  # M

        # Gradient: only the nearest point for each grid point contributes
        # d(val)/d(pts_norm[k]) = sum_j weights[j] * 1[k == min_indices[j]] * d(d_jk)/d(pts_norm[k])
        # d(d_jk)/d(pts_norm[k]) = -grid_pts[j] / sqrt(1 - dot[j,k]^2)
        
        # Build sparse gradient contribution
        grad_pts_norm = np.zeros((N, 3))
        
        # For each grid point j, its nearest point is min_indices[j]
        # Contribution: weights[j] * (-grid_pts[j] / sqrt(1 - dot[j, min_indices[j]]^2))
        denom = np.sqrt(np.maximum(1.0 - dot[np.arange(M), min_indices]**2, 1e-10))  # M
        factors = weights / denom  # M
        
        # Accumulate gradient for each point
        # Use np.add.at for efficient sparse accumulation
        for j in range(M):
            k = min_indices[j]
            grad_pts_norm[k] -= factors[j] * grid_pts[j]

        # Chain rule: d(pts_norm)/d(pts) = (I - pts_norm pts_norm^T) / norm
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200, ftol=1e-10, gtol=1e-8):
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
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
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
            rng = np.random.RandomState(300 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 5:
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 6:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.05 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        else:
            if best_points is not None:
                rng = np.random.RandomState(600 + restart)
                points = best_points + 0.02 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 4.0, 0.1, 0.99, 100)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 20.0
        if time_left > 15.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=300)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 30.0:
        for trial in range(6):
            if time.time() > deadline - 15.0:
                break
            rng = np.random.RandomState(700 + trial)
            noise_scale = 0.05 / (trial + 1)
            candidate = best_points + noise_scale * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 10.0
            if time_left > 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=150, ftol=1e-12, gtol=1e-10)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Coordinate-wise local search
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()
        improved = True
        max_rounds = 6
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 8.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 8.0:
                    break
                
                old_p = points[i].copy()
                best_local_rad = best_rad
                best_local_p = old_p.copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count)
                for trial in range(6):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        rad = compute_covering_radius_exact(points)
                        if rad < best_local_rad - 1e-12:
                            best_local_rad = rad
                            best_local_p = new_p.copy()
                        else:
                            points[i] = old_p
                            break
                    else:
                        continue
                    break
                
                # If we improved, keep the new position
                if best_local_rad < best_rad - 1e-12:
                    points[i] = best_local_p
                    best_rad = best_local_rad
                    best_points = points.copy()
                    improved = True
                else:
                    points[i] = old_p

    # Phase 4: Final polish with L-BFGS-B using very fine grid and tight tolerances
    if best_points is not None and time.time() < deadline - 8.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 4.0
        if time_left > 5.0:
            final, _ = optimize_with_lbfgsb(final, GRID_VERY_FINE, maxiter=50, ftol=1e-14, gtol=1e-12)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
