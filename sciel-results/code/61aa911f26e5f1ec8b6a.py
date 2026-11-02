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

    def fibonacci_spiral(n, offset=0.0):
        i = np.arange(n) + 0.5 + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
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
            
            return float(np.max(radii))
        except Exception:
            return np.inf

    def compute_worst_facet(pts):
        """Return (max_radius, face_indices, circumcenter) for the worst facet."""
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
            cc_arr = np.zeros((len(faces), 3))
            
            if np.any(valid):
                cc_valid = normal[valid] / norms[valid, np.newaxis]
                centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                dots = np.sum(cc_valid * centroids, axis=1)
                cc_valid[dots < 0] *= -1
                cos_r = np.clip(np.sum(cc_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                radii[valid] = r
                cc_arr[valid] = cc_valid
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                mid = (a_deg + b_deg + c_deg) / 3.0
                mid_norms = np.linalg.norm(mid, axis=1, keepdims=True)
                mid_norms[mid_norms < 1e-10] = 1.0
                cc_arr[~valid] = mid / mid_norms
            
            max_idx = np.argmax(radii)
            return float(radii[max_idx]), faces[max_idx], cc_arr[max_idx]
        except Exception:
            return np.inf, np.array([0, 1, 2]), np.array([0.0, 0.0, 1.0])

    def make_grid(n_theta=100, n_phi=200):
        """Create a dense grid of points on the unit sphere."""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_points = make_grid(100, 200)  # ~20000 points

    def soft_max_objective(flat_pts, beta=80.0):
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
        
        weighted_coeff = outer_weights[:, np.newaxis] * inner_weights  # shape (G, N)
        coeff_div_sqrt = weighted_coeff / sqrt_oms  # shape (G, N)
        
        term1 = grid_points.T @ coeff_div_sqrt  # shape (3, N)
        term2_scalar = np.sum(coeff_div_sqrt * dots, axis=0)  # shape (N,)
        
        grad_pts = -(term1.T - term2_scalar[:, np.newaxis] * pts)
        
        # Project to tangent space
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
        grad_flat = grad_tangent.ravel()
        
        return objective, grad_flat

    def targeted_refinement(points, max_iters=500, time_budget=None):
        """Refine by targeting the worst facet: project vertices toward antipode of circumcenter."""
        if time_budget is None:
            time_budget = deadline - 5.0
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            rad, face, cc = compute_worst_facet(pts)
            if not np.isfinite(rad):
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            # Antipode of circumcenter
            antipode = -cc
            antipode_norm = np.linalg.norm(antipode)
            if antipode_norm < 1e-10:
                break
            antipode /= antipode_norm
            
            improved = False
            
            # Try multiple step sizes toward antipode
            for step in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
                new_a = a + step * antipode
                new_b = b + step * antipode
                new_c = c + step * antipode
                
                na = np.linalg.norm(new_a)
                nb = np.linalg.norm(new_b)
                nc = np.linalg.norm(new_c)
                if na < 1e-10 or nb < 1e-10 or nc < 1e-10:
                    continue
                
                new_a /= na
                new_b /= nb
                new_c /= nc
                
                pts[i] = new_a
                pts[j] = new_b
                pts[k] = new_c
                
                new_rad, _, _ = compute_worst_facet(pts)
                if new_rad < rad:
                    improved = True
                    break
                else:
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
            
            if not improved:
                # Try moving vertices individually toward antipode
                for idx, p in [(i, a), (j, b), (k, c)]:
                    for step in [0.01, 0.005, 0.002, 0.001]:
                        new_p = p + step * antipode
                        np_norm = np.linalg.norm(new_p)
                        if np_norm < 1e-10:
                            continue
                        new_p /= np_norm
                        pts[idx] = new_p
                        new_rad, _, _ = compute_worst_facet(pts)
                        if new_rad < rad:
                            improved = True
                            break
                        else:
                            pts[idx] = p
                    if improved:
                        break
            
            if not improved:
                # Try small random perturbations in tangent plane to escape local minima
                rng = np.random.RandomState(it + 42)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    for _ in range(8):
                        rand_vec = rng.randn(3)
                        tangent = rand_vec - np.dot(rand_vec, p) * p
                        t_norm = np.linalg.norm(tangent)
                        if t_norm < 1e-10:
                            continue
                        tangent /= t_norm
                        for step in [0.005, 0.002, 0.001]:
                            new_p = p + step * tangent
                            np_norm = np.linalg.norm(new_p)
                            if np_norm < 1e-10:
                                continue
                            new_p /= np_norm
                            pts[idx] = new_p
                            new_rad, _, _ = compute_worst_facet(pts)
                            if new_rad < rad:
                                improved = True
                                break
                            else:
                                pts[idx] = p
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return pts

    def newton_polish(points, max_iters=50):
        """High-precision polish: solve for the worst facet to have all three vertices at equal distance from circumcenter."""
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            rad, face, cc = compute_worst_facet(pts)
            if not np.isfinite(rad):
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            # Check if all three vertices are at equal distance from circumcenter
            cos_a = np.clip(np.dot(cc, a), -1.0, 1.0)
            cos_b = np.clip(np.dot(cc, b), -1.0, 1.0)
            cos_c = np.clip(np.dot(cc, c), -1.0, 1.0)
            d_a = np.arccos(cos_a)
            d_b = np.arccos(cos_b)
            d_c = np.arccos(cos_c)
            
            # If all distances are within tolerance, we're converged
            if max(d_a, d_b, d_c) - min(d_a, d_b, d_c) < 1e-12:
                break
            
            # Move the farthest vertex toward the circumcenter's antipode
            dists = [d_a, d_b, d_c]
            idxs = [i, j, k]
            pts_list = [a, b, c]
            
            max_d = max(dists)
            min_d = min(dists)
            
            # Move vertex with max distance toward antipode of cc
            max_idx = idxs[np.argmax(dists)]
            p = pts[max_idx].copy()
            
            antipode = -cc
            antipode_norm = np.linalg.norm(antipode)
            if antipode_norm < 1e-10:
                break
            antipode /= antipode_norm
            
            # Binary search for step size that equalizes distances
            lo, hi = 0.0, 0.05
            best_step = 0.0
            best_diff = np.inf
            
            for _ in range(30):
                mid = (lo + hi) / 2
                new_p = p + mid * antipode
                np_norm = np.linalg.norm(new_p)
                if np_norm < 1e-10:
                    hi = mid
                    continue
                new_p /= np_norm
                
                new_cos = np.clip(np.dot(cc, new_p), -1.0, 1.0)
                new_d = np.arccos(new_cos)
                new_diff = abs(new_d - min_d)
                
                if new_diff < best_diff:
                    best_diff = new_diff
                    best_step = mid
                
                # Check if moving this vertex reduces the overall max
                pts[max_idx] = new_p
                new_rad, _, _ = compute_worst_facet(pts)
                if new_rad < rad:
                    lo = mid
                else:
                    pts[max_idx] = p
                    hi = mid
            
            if best_step > 1e-10:
                new_p = p + best_step * antipode
                np_norm = np.linalg.norm(new_p)
                if np_norm > 1e-10:
                    new_p /= np_norm
                    pts[max_idx] = new_p
            else:
                break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multi-start with L-BFGS-B using soft-max surrogate
    rng = np.random.RandomState(12345)
    
    for start_iter in range(3):
        if time.time() > deadline - 60.0:
            break
        
        if start_iter == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif start_iter == 1:
            points = fibonacci_spiral(N, offset=0.3).astype(np.float64)
            # Add small random perturbation
            perturb = rng.randn(N, 3) * 0.01
            points = points + perturb
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points = points / norms
        else:
            # Random initialization
            points = rng.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points = points / norms
        
        x0 = points.ravel()
        beta = 80.0
        
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
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300}
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
        best_points = fibonacci_spiral(N).astype(np.float64)
        best_rad = compute_covering_radius_hull(best_points)

    # Phase 2: Targeted refinement
    if time.time() < deadline - 15.0:
        refined = targeted_refinement(best_points.copy(), max_iters=800, time_budget=deadline - 15.0)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    # Phase 3: Newton polish
    if time.time() < deadline - 5.0:
        polished = newton_polish(best_points.copy(), max_iters=100)
        rad = compute_covering_radius_hull(polished)
        if rad < best_rad:
            best_rad = rad
            best_points = polished.copy()

    # Phase 4: Final targeted refinement with tighter budget
    if time.time() < deadline - 3.0:
        refined = targeted_refinement(best_points.copy(), max_iters=300, time_budget=deadline - 3.0)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    return best_points

# EVOLVE-BLOCK-END
