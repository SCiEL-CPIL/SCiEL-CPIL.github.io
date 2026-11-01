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
GRID_VERY_FINE = _make_fib_grid(16000)

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

    def compute_covering_radius_quick(pts, grid_pts=None):
        """Fast grid-based estimate of covering radius."""
        if grid_pts is None:
            grid_pts = GRID_COARSE
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

    def grid_objective_and_grad(x_flat, grid_pts):
        """
        Fast smooth objective: weighted sum of squared angular distances from grid points
        to their nearest configuration point, plus a max-like term.
        x_flat: flattened (N,3) array of points.
        grid_pts: precomputed grid of sphere points.
        Returns (value, gradient).
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute dot products: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        
        # Angular distances: M x N
        ang_dists = np.arccos(dot)  # M x N
        
        # For each grid point, find the min distance to any configuration point
        min_dists = np.min(ang_dists, axis=1)  # M
        
        # Use a robust smooth max approximation: 
        # val = max_d + (1/T) * log(sum exp(T * (d_j - max_d)))
        T = 30.0
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(shifted * T)
        sum_exp = np.sum(exp_shifted)
        val = max_d + np.log(sum_exp) / T
        
        # Gradient: weighted average of gradients of d_j
        # w_j = exp(T * (d_j - max_d)) / sum_exp
        w_j = exp_shifted / sum_exp  # M
        
        # d(d_j)/d(pts_norm[k]) where d_j = min_k ang_dists[j,k]
        # For the argmin k_j, d(d_j)/d(pts_norm[k_j]) = -grid_pts[j] / sqrt(1 - dot[j,k_j]^2)
        # For smoothness, use soft-min weights
        T_min = 30.0
        exp_neg = np.exp(-T_min * ang_dists)  # M x N
        sum_exp_neg = np.sum(exp_neg, axis=1)  # M
        w_jk = exp_neg / sum_exp_neg[:, np.newaxis]  # M x N
        
        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10))  # M x N
        factor = (w_j[:, np.newaxis] * w_jk) / denom  # M x N
        grad_pts_norm = -(factor.T @ grid_pts)  # N x 3
        
        # Chain rule for normalization
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
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
            options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def simulated_annealing_local(points, current_rad, max_iters=2000):
        """Simulated annealing to escape local minima."""
        T_start = 0.05
        T_end = 0.001
        best_local = points.copy()
        best_local_rad = current_rad
        current = points.copy()
        current_rad = current_rad
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Pick a random point to move
            idx = np.random.randint(N)
            old_p = current[idx].copy()
            
            # Random perturbation
            noise = np.random.randn(3)
            noise -= np.dot(noise, old_p) * old_p
            noise_norm = np.linalg.norm(noise)
            if noise_norm < 1e-10:
                continue
            noise /= noise_norm
            
            step_size = 0.02 * (1.0 + T * 10)
            new_p = old_p + step_size * noise
            new_p /= np.linalg.norm(new_p)
            
            current[idx] = new_p
            new_rad = compute_covering_radius_quick(current)
            
            delta = new_rad - current_rad
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                current_rad = new_rad
                if new_rad < best_local_rad:
                    best_local_rad = new_rad
                    best_local = current.copy()
            else:
                current[idx] = old_p
        
        return best_local, best_local_rad

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
            rng = np.random.RandomState(300 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            # Add some structure: pair up opposite points
            for i in range(0, N-1, 2):
                points[i+1] = -points[i]
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        # Use higher exponent for better separation
        points = gradient_descent_repulsion(points, 4.0, 0.05, 0.98, 150)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=200)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Simulated annealing to escape local minima
    if best_points is not None and time.time() < deadline - 30.0:
        sa_points, sa_rad = simulated_annealing_local(best_points, best_rad, max_iters=3000)
        if sa_rad < best_rad:
            # Verify with exact computation
            exact_rad = compute_covering_radius_exact(sa_points)
            if exact_rad < best_rad:
                best_rad = exact_rad
                best_points = sa_points.copy()

    # Phase 3: Local refinement with L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 25.0:
        for trial in range(4):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.03 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 8.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=150)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 4: Fast local search with quick grid check
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()
        improved = True
        max_rounds = 15
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 8.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                old_p = points[i].copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count * 100)
                for trial in range(3):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    # Adaptive step sizes
                    for step in [0.03, 0.015, 0.008]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        # Quick check with grid first
                        quick_rad = compute_covering_radius_quick(points)
                        if quick_rad < best_rad - 0.0001:
                            # Verify with exact hull
                            rad = compute_covering_radius_exact(points)
                            if rad < best_rad - 1e-10:
                                best_rad = rad
                                best_points = points.copy()
                                improved = True
                                old_p = new_p.copy()
                                break
                            else:
                                points[i] = old_p
                        else:
                            points[i] = old_p
                    else:
                        continue
                    break

    # Phase 5: Final polish with L-BFGS-B using very fine grid
    if best_points is not None and time.time() < deadline - 8.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 5.0
        if time_left > 8.0:
            max_iter = 150 if time_left > 25.0 else 80
            final, _ = optimize_with_lbfgsb(final, GRID_VERY_FINE, maxiter=max_iter)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final
        elif time_left > 5.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=50)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
