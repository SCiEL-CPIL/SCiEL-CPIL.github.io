"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
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
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
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

    def compute_face_radii(pts):
        """Return array of face circumradii (in radians) for gradient-aware optimization."""
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

            return radii, faces
        except Exception:
            return None, None

    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps."""
        for _ in range(n_iter):
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    def rbf_coverage_optimize(pts_init, grid_n=60, sigma=0.15, max_iter=200):
        """
        Differentiable min-max optimization using Gaussian RBF coverage field.
        Parameterize points via SO(3) rotation + radial perturbation for smooth gradients.
        """
        # Build a dense spherical grid for the coverage field
        n_theta = grid_n
        n_phi = 2 * grid_n
        theta = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / n_theta
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        grid_x = np.sin(TH) * np.cos(PH)
        grid_y = np.sin(TH) * np.sin(PH)
        grid_z = np.cos(TH)
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        n_grid = len(grid_pts)

        # Precompute grid outer products for efficient dot products
        # grid_pts: (n_grid, 3), pts: (N, 3)
        # dot[i,j] = grid_pts[i] @ pts[j]

        def objective_and_grad(flat_params):
            """
            flat_params: (N, 4) -> first 3 are direction, 4th is unused (or use 3 params)
            Actually, let's use a different parameterization:
            We'll optimize the N*3 coordinates directly but project to sphere.
            For gradient computation via RBF, we need the angular distance.
            
            Better approach: use the fact that for a Gaussian kernel K(d) = exp(-d^2/(2*sigma^2)),
            where d is angular distance, the gradient w.r.t. a point p_i is:
            dK/dd * d(d)/d(p_i)
            
            For a grid point g and site p:
            d = arccos(g·p)
            K = exp(-d^2/(2*sigma^2))
            coverage(g) = max_i K_i(g)  [or soft-max]
            
            We want to minimize max_g coverage(g).
            
            Let's use a soft-max: F = log(sum_g exp(beta * coverage(g))) / beta
            Then dF/dp_i involves the gradient of the max coverage.
            
            Actually, simpler: minimize sum_g exp(beta * coverage(g)) or just the max.
            For differentiability, use: F = (1/n_grid) * sum_g coverage(g)^p  for large p,
            or F = log(sum_g exp(beta * coverage(g))) / beta
            """
            pts = flat_params.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms

            # Compute angular distances from grid to each point
            # dot: (n_grid, N)
            dot = grid_pts @ pts.T  # (n_grid, N)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dist = np.arccos(dot)  # (n_grid, N)

            # RBF kernel values
            K = np.exp(-ang_dist ** 2 / (2.0 * sigma ** 2))  # (n_grid, N)

            # Coverage at each grid point = max over sites (use soft-max for smoothness)
            beta_cov = 30.0
            max_K = np.max(K, axis=1, keepdims=True)
            # Soft max
            shifted = K - max_K
            soft_max_K = max_K.squeeze() + np.log(np.sum(np.exp(beta_cov * shifted), axis=1)) / beta_cov
            # soft_max_K: (n_grid,)

            # Objective: mean of soft-max coverage (want to minimize, but coverage is "bad" when high)
            # Actually we want to MINIMIZE the maximum angular distance, which corresponds to
            # MINIMIZING the maximum coverage value... wait, that's wrong.
            # 
            # Let me reconsider. The covering radius is the max angular distance from any grid point
            # to its nearest site. We want to minimize this.
            #
            # Let min_dist(g) = min_i ang_dist(g, p_i)
            # We want to minimize max_g min_dist(g)
            #
            # Using RBF: define "uncoveredness" U(g) = exp(min_dist(g)^2 / (2*sigma^2))
            # Then minimizing max_g U(g) is equivalent to minimizing max_g min_dist(g).
            #
            # For smoothness: F = log(sum_g exp(beta * U(g))) / beta

            min_dist = np.min(ang_dist, axis=1)  # (n_grid,)
            U = np.exp(min_dist ** 2 / (2.0 * sigma ** 2))  # (n_grid,)

            # Soft-max surrogate
            beta_obj = 20.0
            max_U = np.max(U)
            shifted_U = U - max_U
            log_sum = max_U + np.log(np.sum(np.exp(beta_obj * shifted_U))) / beta_obj
            obj = log_sum

            # Gradient computation
            # dU_g/dp_i where U_g = exp(min_dist(g)^2 / (2*sigma^2))
            # Only the nearest site contributes (in the hard max case)
            # For soft handling, weight by which site is "responsible"
            
            # Find nearest site for each grid point
            nearest = np.argmin(ang_dist, axis=1)  # (n_grid,)
            min_dist_vals = min_dist  # (n_grid,)
            
            # d(min_dist)/d(p_i) for the nearest site i:
            # min_dist = arccos(g · p_i)
            # d(arccos(x))/dx = -1/sqrt(1-x^2)
            # d(g·p_i)/d(p_i) = g
            # So d(min_dist)/d(p_i) = -g / sqrt(1 - (g·p_i)^2)
            # But we need the tangential component (projection onto sphere)
            
            grad = np.zeros((N, 3))
            
            for gi in range(n_grid):
                gi_idx = nearest[gi]
                g = grid_pts[gi]
                p = pts[gi_idx]
                d = min_dist_vals[gi]
                cos_d = np.dot(g, p)
                sin_d = np.sqrt(max(1.0 - cos_d * cos_d, 1e-15))
                
                if sin_d < 1e-10:
                    continue
                    
                # dU/dp = U * (min_dist / sigma^2) * d(min_dist)/dp
                # d(min_dist)/dp = -g / sin_d (radial component in parameter space)
                # But p is on the sphere, so we need the tangential gradient
                # The gradient on the sphere: project -g/sin_d to be tangent to p
                # tangent_grad = (-g/sin_d) - dot(-g/sin_d, p)*p
                # = (-g/sin_d) + cos_d/sin_d * p
                
                dU_dp = U[gi] * (d / (sigma ** 2)) * (-g / sin_d + (cos_d / sin_d) * p)
                
                # Weight by soft-max weight
                w_g = np.exp(beta_obj * (U[gi] - max_U)) / np.sum(np.exp(beta_obj * (U - max_U)))
                
                grad[gi_idx] += w_g * dU_dp
            
            # Normalize gradient
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-15] = 1.0
            grad = grad / grad_norms
            
            return obj, grad.ravel()

        # Initial parameters
        x0 = pts_init.ravel().copy()

        # Use L-BFGS-B
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 20}
            )
            pts_opt = result.x.reshape(N, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts_opt = pts_opt / norms
            return pts_opt
        except Exception:
            return pts_init

    def rbf_optimize_vectorized(pts_init, grid_n=50, sigma=0.12, max_iter=100):
        """
        Vectorized RBF coverage optimization with proper gradient computation.
        Uses a finer grid and more careful gradient handling.
        """
        # Build spherical grid
        n_theta = grid_n
        n_phi = 2 * grid_n
        theta = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / n_theta
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        grid_x = np.sin(TH) * np.cos(PH)
        grid_y = np.sin(TH) * np.sin(PH)
        grid_z = np.cos(TH)
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        n_grid = len(grid_pts)

        def obj_grad(flat_params):
            pts = flat_params.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms

            # Angular distances: (n_grid, N)
            dot = grid_pts @ pts.T
            dot = np.clip(dot, -1.0, 1.0)
            ang_dist = np.arccos(dot)

            # Min distance from each grid point to nearest site
            min_dist, nearest = np.min(ang_dist, axis=1), np.argmin(ang_dist, axis=1)

            # Uncoveredness
            U = np.exp(min_dist ** 2 / (2.0 * sigma ** 2))

            # Soft-max objective
            beta_obj = 25.0
            max_U = np.max(U)
            shifted_U = U - max_U
            exp_shifted = np.exp(beta_obj * shifted_U)
            sum_exp = np.sum(exp_shifted)
            obj = max_U + np.log(sum_exp) / beta_obj

            # Weights for each grid point
            weights = exp_shifted / sum_exp  # (n_grid,)

            # Gradient: for each grid point, the gradient flows to its nearest site
            # dU_g/dp_i = U_g * (min_dist_g / sigma^2) * d(min_dist_g)/d(p_i)
            # d(min_dist)/d(p_i) on sphere: tangent component of -g/sin(d)
            grad = np.zeros((N, 3))

            # Vectorized: for each grid point, compute contribution to nearest site
            g = grid_pts  # (n_grid, 3)
            p_nearest = pts[nearest]  # (n_grid, 3)
            cos_d = np.clip(np.sum(g * p_nearest, axis=1), -1.0, 1.0)
            sin_d = np.sqrt(np.maximum(1.0 - cos_d * cos_d, 1e-15))

            # Tangent gradient direction: -g/sin_d + (cos_d/sin_d)*p
            # This is the direction on the sphere that increases min_dist
            # We want to DECREASE min_dist, so we move in the negative of this
            # dU/dp = U * (d/sigma^2) * tangent_dir
            # where tangent_dir points in direction of increasing d
            # So to decrease U, we move opposite: -tangent_dir

            d_factor = U * (min_dist / (sigma ** 2)) / sin_d  # (n_grid,)
            # tangent_dir = -g/sin_d + (cos_d/sin_d)*p
            # Contribution to gradient (we want to decrease obj, so negate)
            contrib = -d_factor[:, np.newaxis] * (g / sin_d[:, np.newaxis] - (cos_d / sin_d)[:, np.newaxis] * p_nearest)

            # Weighted sum
            weighted_contrib = weights[:, np.newaxis] * contrib  # (n_grid, 3)

            # Scatter to nearest sites
            for gi in range(n_grid):
                idx = nearest[gi]
                grad[idx] += weighted_contrib[gi]

            return obj, grad.ravel()

        x0 = pts_init.ravel().copy()
        try:
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 1e-12, 'maxls': 20}
            )
            pts_opt = result.x.reshape(N, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts_opt = pts_opt / norms
            return pts_opt
        except Exception:
            return pts_init

    def targeted_face_reduction(points, n_iter=100):
        """Directly reduce the largest facet radii by moving vertices toward circumcenters."""
        best = points.copy()
        best_rad = compute_covering_radius(best)

        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break

            face_radii, faces = compute_face_radii(points)
            if face_radii is None:
                break

            top_k = min(15, len(faces))
            top_indices = np.argsort(face_radii)[-top_k:]

            displacement = np.zeros_like(points)
            for fi in top_indices:
                if face_radii[fi] < best_rad * 0.7:
                    continue
                f = faces[fi]
                a, b, c = points[f[0]], points[f[1]], points[f[2]]
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norm_c = np.linalg.norm(c_dir)
                if norm_c < 1e-10:
                    continue
                c_dir = c_dir / norm_c

                # Move vertices TOWARD circumcenter (reduces circumradius)
                for vi in f:
                    p = points[vi]
                    tangent = c_dir - np.dot(p, c_dir) * c_dir
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 1e-10:
                        displacement[vi] += tangent / tangent_norm

            norm_disp = np.linalg.norm(displacement, axis=1, keepdims=True)
            norm_disp[norm_disp < 1e-15] = 1.0
            step_size = 0.015 * (1.0 - it / max(n_iter, 1))
            points = points + step_size * displacement / norm_disp
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

            rad = compute_covering_radius(points)
            if rad < best_rad - 1e-10:
                best_rad = rad
                best = points.copy()

        return best

    def lloyd_refine(points, n_iter=30):
        """Lloyd-based refinement using face circumcenters as cell centroids."""
        best = points.copy()
        best_rad = compute_covering_radius(best)

        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break

            face_radii, faces = compute_face_radii(points)
            if face_radii is None:
                break

            centroids = np.zeros_like(points)
            counts = np.zeros(len(points))

            for fi in range(len(faces)):
                f = faces[fi]
                a, b, c = points[f[0]], points[f[1]], points[f[2]]
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norm_c = np.linalg.norm(c_dir)
                if norm_c < 1e-10:
                    continue
                c_dir = c_dir / norm_c

                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                weight = d_ab * d_bc * d_ca + 1e-10

                for vi in f:
                    centroids[vi] += weight * c_dir
                    counts[vi] += weight

            for i in range(len(points)):
                if counts[i] > 1e-10:
                    centroid_dir = centroids[i] / counts[i]
                    cn = np.linalg.norm(centroid_dir)
                    if cn > 1e-10:
                        centroid_dir = centroid_dir / cn
                        alpha = 0.25 * (1.0 - it / n_iter)
                        points[i] = points[i] + alpha * (centroid_dir - points[i])
                        points[i] = points[i] / np.linalg.norm(points[i])

            rad = compute_covering_radius(points)
            if rad < best_rad - 1e-12:
                best_rad = rad
                best = points.copy()

        return best

    best_points = None
    best_rad = np.inf

    # Strategy: multiple restarts with RBF-based optimization
    n_restarts = 4

    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                points = best_points + 0.1 * np.random.randn(N, 3)
            else:
                points = fibonacci_spiral(N)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        # Quick repulsion pre-pass to get out of bad local minima
        alpha = 0.03
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= 0.95

        # RBF-based optimization (the main structural change)
        if time.time() < deadline - 20.0:
            rbf_result = rbf_optimize_vectorized(points.copy(), grid_n=40, sigma=0.12, max_iter=80)
            rad_rbf = compute_covering_radius(rbf_result)
            if rad_rbf < best_rad:
                best_rad = rad_rbf
                best_points = rbf_result

        # Lloyd refinement
        if time.time() < deadline - 15.0 and best_points is not None:
            lloyd_result = lloyd_refine(best_points.copy(), n_iter=25)
            rad_lloyd = compute_covering_radius(lloyd_result)
            if rad_lloyd < best_rad:
                best_rad = rad_lloyd
                best_points = lloyd_result

        # Targeted face reduction
        if time.time() < deadline - 10.0 and best_points is not None:
            targeted = targeted_face_reduction(best_points.copy(), n_iter=50)
            rad_t = compute_covering_radius(targeted)
            if rad_t < best_rad:
                best_rad = rad_t
                best_points = targeted

        # Another RBF pass with different parameters
        if time.time() < deadline - 10.0 and best_points is not None:
            rbf2 = rbf_optimize_vectorized(best_points.copy(), grid_n=50, sigma=0.08, max_iter=60)
            rad_rbf2 = compute_covering_radius(rbf2)
            if rad_rbf2 < best_rad:
                best_rad = rad_rbf2
                best_points = rbf2

    # Final polish
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.003, 40)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

        if time.time() < deadline - 3.0:
            final2 = targeted_face_reduction(best_points.copy(), n_iter=30)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
