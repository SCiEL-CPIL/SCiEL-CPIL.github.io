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

    def repulsion_energy_and_grad(pts, exponent=2.0):
        """
        Compute repulsion energy and gradient.
        Energy: sum_{i<j} 1 / ||x_i - x_j||^{2*exponent}
        This is a smooth surrogate that encourages uniform distribution.
        """
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        # w_ij = 1 / dist_sq^exponent
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        energy = 0.5 * np.sum(w)
        
        # Gradient: d/dx_i sum_{j!=i} (2 - 2 x_i.x_j)^(-exponent)
        # = sum_{j!=i} (-exponent) (2 - 2 x_i.x_j)^(-exponent - 1) * (-2 (x_j - x_i))
        # = 2 * exponent * sum_{j!=i} (2 - 2 x_i.x_j)^(-exponent - 1) * (x_j - x_i)
        # Note: The term (x_j - x_i) is the vector from i to j in Euclidean space, 
        # but on the sphere, we project the gradient to the tangent space.
        
        # Let's compute the raw Euclidean gradient first.
        # w_grad_ij = exponent * (dist_sq)^(-exponent - 1)
        w_grad = exponent * (dist_sq ** (-exponent - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        # grad_i = 2 * sum_j w_grad_ij * (x_j - x_i)
        # = 2 * (sum_j w_grad_ij * x_j) - 2 * x_i * sum_j w_grad_ij
        sum_w_grad = np.sum(w_grad, axis=1)
        sum_w_grad_pts = w_grad @ pts
        
        grad = 2.0 * (sum_w_grad_pts - sum_w_grad[:, np.newaxis] * pts)
        
        # Project gradient to tangent space of sphere
        # grad_tangent = grad - (grad . x_i) x_i
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        return energy, grad_tangent

    def gradient_descent_repulsion(points, exponent, alpha_init, decay, max_iter):
        """Optimize points via gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_and_grad(points, exponent)
            points = points + alpha * grad
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
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
    n_restarts = 5
    exponents = [1.5, 2.0, 2.5, 3.0, 2.0]
    
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
        elif restart == 3:
            rng = np.random.RandomState(200)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent_repulsion(points, exp, 0.05, 0.995, 100)
        rad = compute_covering_radius_hull(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: L-BFGS-B optimization on repulsion energy
    if best_points is not None and time.time() < deadline - 30.0:
        # Use a higher exponent for sharper repulsion in the final polish
        for exponent in [2.0, 3.0]:
            if time.time() > deadline - 15.0:
                break
            
            x0 = best_points.ravel()
            
            def objective_wrapper(x):
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                pts = pts / norms
                val, _ = repulsion_energy_and_grad(pts, exponent)
                return val
            
            def gradient_wrapper(x):
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                pts = pts / norms
                _, grad = repulsion_energy_and_grad(pts, exponent)
                return grad.ravel()
            
            try:
                result = minimize(
                    objective_wrapper,
                    x0,
                    jac=gradient_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
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

    # Phase 3: Targeted refinement on the worst facet
    if best_points is not None and time.time() < deadline - 10.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 4: Final L-BFGS-B polish with higher exponent
    if best_points is not None and time.time() < deadline - 8.0:
        exponent = 4.0
        x0 = best_points.ravel()
        
        def objective_wrapper(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            val, _ = repulsion_energy_and_grad(pts, exponent)
            return val
        
        def gradient_wrapper(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            _, grad = repulsion_energy_and_grad(pts, exponent)
            return grad.ravel()
        
        try:
            result = minimize(
                objective_wrapper,
                x0,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300}
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
