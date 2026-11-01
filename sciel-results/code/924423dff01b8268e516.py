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

    def facet_softmax_objective_and_grad(pts, temperature=50.0):
        """
        Soft-max surrogate for the covering radius based on convex hull facets.
        For each facet (triangle on the convex hull), compute the circumradius.
        Use log-sum-exp to create a smooth approximation of the max circumradius.
        
        The gradient moves each vertex toward its facet's circumcenter,
        which shrinks the circumradius of that facet.
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            n_faces = len(faces)
            
            if n_faces == 0:
                return 0.0, np.zeros_like(pts)
            
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumradius for each facet
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(n_faces)
            circumcenters = np.zeros((n_faces, 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = c_dir_valid
            
            if np.any(~valid):
                # Degenerate case: use max pairwise distance
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # Approximate circumcenter as average
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                circumcenters[~valid] /= np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
            
            # Soft-max of radii
            max_r = np.max(radii)
            if max_r < 1e-10:
                return 0.0, np.zeros_like(pts)
            
            shifted = radii - max_r
            exp_shifted = np.exp(shifted * temperature)
            sum_exp = np.sum(exp_shifted)
            val = max_r + np.log(sum_exp) / temperature
            
            # Weights for each facet
            w = exp_shifted / sum_exp  # n_faces
            
            # Gradient: for each facet, move vertices toward circumcenter
            # d(val)/d(pts[i]) = sum over facets f containing i of w[f] * (circumcenter[f] - pts[i]) projected to tangent
            grad = np.zeros_like(pts)
            
            # For each facet, add contribution to its three vertices
            for f in range(n_faces):
                if w[f] < 1e-10:
                    continue
                cc = circumcenters[f]
                for vi, vertex_idx in enumerate([faces[f, 0], faces[f, 1], faces[f, 2]]):
                    p = pts[vertex_idx]
                    # Project (cc - p) to tangent space at p
                    tangent = cc - p
                    tangent -= np.dot(tangent, p) * p
                    grad[vertex_idx] += w[f] * tangent
            
            return float(val), grad

        except Exception:
            # Fallback to grid-based soft-max
            dot = np.dot(GRID_COARSE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            max_d = np.max(min_dists)
            T = 20.0
            shifted = min_dists - max_d
            exp_shifted = np.exp(shifted * T)
            sum_exp = np.sum(exp_shifted)
            val = max_d + np.log(sum_exp) / T
            w_j = exp_shifted / sum_exp
            nearest_idx = np.argmin(ang_dists, axis=1)
            grad = np.zeros_like(pts)
            for j in range(len(GRID_COARSE)):
                k = nearest_idx[j]
                if w_j[j] > 1e-10:
                    d = dot[j, k]
                    denom = np.sqrt(max(1.0 - d*d, 1e-10))
                    grad[k] += w_j[j] * (-GRID_COARSE[j] / denom)
            # Project to tangent space
            for i in range(N):
                p = pts[i]
                grad[i] -= np.dot(grad[i], p) * p
            return float(val), grad

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
            if time.time() > deadline - 10.0:
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

    def optimize_facet_softmax(pts, temperature=50.0, max_iter=100):
        """
        Optimize using gradient descent on the facet soft-max objective.
        This directly targets the covering radius by shrinking facet circumradii.
        """
        alpha = 0.05
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            val, grad = facet_softmax_objective_and_grad(pts, temperature)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            # Decay learning rate
            alpha *= 0.98
            
            # Occasionally add repulsion to maintain uniformity
            if it % 10 == 5:
                _, rep_grad = repulsion_energy_grad(pts, 2.0)
                rep_norm = np.linalg.norm(rep_grad)
                if rep_norm > 1e-10:
                    rep_grad = rep_grad / rep_norm
                pts = pts + 0.01 * rep_grad
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        return pts

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            pts_norm = pts / norms
            val, _ = facet_softmax_objective_and_grad(pts_norm, temperature=30.0)
            return val
        
        def gradient(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            pts_norm = pts / norms
            _, g = facet_softmax_objective_and_grad(pts_norm, temperature=30.0)
            # Project to tangent space
            for i in range(N):
                p = pts_norm[i]
                g[i] -= np.dot(g[i], p) * p
            return g.flatten()
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 300, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + facet optimization
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
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.05 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 3.0, 0.08, 0.985, 200)
        
        # Then optimize with facet soft-max gradient descent
        time_left = deadline - time.time() - 20.0
        if time_left > 15.0:
            points = optimize_facet_softmax(points, temperature=50.0, max_iter=80)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with L-BFGS-B on facet objective
    if best_points is not None and time.time() < deadline - 25.0:
        for trial in range(3):
            if time.time() > deadline - 15.0:
                break
            rng = np.random.RandomState(600 + trial)
            candidate = best_points + 0.02 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 10.0
            if time_left > 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_COARSE, maxiter=60)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Local search with direct facet circumradius minimization
    if best_points is not None and time.time() < deadline - 10.0:
        points = best_points.copy()
        improved = True
        max_rounds = 10
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 5.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                old_p = points[i].copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count * 100)
                for trial in range(5):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    # Adaptive step sizes
                    for step in [0.02, 0.01, 0.005, 0.002]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        # Compute exact covering radius
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
                        continue
                    break

    # Phase 4: Final polish with facet gradient descent + local search
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 5.0:
            # Use facet gradient descent for final polish
            max_iter = 60 if time_left > 20.0 else 30
            final = optimize_facet_softmax(final, temperature=30.0, max_iter=max_iter)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final.copy()
            
            # Additional local search on the polished result
            if time.time() < deadline - 5.0:
                points = best_points.copy()
                rng = np.random.RandomState(999)
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    old_p = points[i].copy()
                    for step in [0.01, 0.005, 0.002]:
                        direction = rng.randn(3)
                        direction -= np.dot(direction, old_p) * old_p
                        dir_norm = np.linalg.norm(direction)
                        if dir_norm < 1e-10:
                            continue
                        direction /= dir_norm
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-10:
                            best_rad = rad
                            best_points = points.copy()
                            old_p = new_p.copy()
                        else:
                            points[i] = old_p

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
