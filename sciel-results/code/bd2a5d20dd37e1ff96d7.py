"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    # Precompute a high-resolution grid on the sphere
    n_lat = 100
    n_lon = 200
    lat = np.linspace(-np.pi/2 + 0.5*np.pi/n_lat, np.pi/2 - 0.5*np.pi/n_lat, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    gz = np.sin(LA)
    gx = np.cos(LA) * np.cos(LO)
    gy = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    grid_pts = grid_pts / np.linalg.norm(grid_pts, axis=1, keepdims=True)
    n_grid = len(grid_pts)

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius using convex hull face circumradii."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            # For each triangle, compute the circumcenter on the sphere
            # The covering radius for a face is the max angular distance from the circumcenter to the vertices
            # If the circumcenter is on the opposite side, use the largest edge
            radii = np.zeros(len(faces))

            for idx in range(len(faces)):
                ai, bi, ci = a[idx], b[idx], c[idx]
                # Circumcenter of spherical triangle
                # Using the formula: circumcenter is proportional to (a x b) x (b x c) + ... 
                # Actually, for a spherical triangle, the circumcenter C satisfies:
                # C . a = C . b = C . c = cos(r)
                # We can solve this linear system
                A = np.array([ai, bi, ci])  # (3,3)
                # C = A^T * (A A^T)^{-1} * [1,1,1]^T / ||...||
                # But this is for the plane. For sphere, we need:
                # The circumcenter direction is given by:
                # C = (a × b) × (b × c) + ... no
                # Let's use: C is in the plane spanned by a,b,c and equidistant from all three
                # C = α*a + β*b + γ*c where C.a = C.b = C.c
                # This gives a linear system
                
                # Alternative: use the formula for spherical circumcenter
                # The circumcenter of a spherical triangle with vertices a,b,c (unit vectors)
                # is proportional to:
                # (a × b) × (b × c) + (b × c) × (c × a) + (c × a) × (a × b)
                # Wait, that's the incenter direction? No.
                
                # Let me use a different approach: 
                # The circumcenter C on the sphere satisfies C·a = C·b = C·c
                # C = t1*a + t2*b + t3*c
                # C·a = t1 + t2*(b·a) + t3*(c·a) = cos_r
                # C·b = t1*(a·b) + t2 + t3*(c·b) = cos_r
                # C·c = t1*(a·c) + t2*(b·c) + t3 = cos_r
                # Subtracting: (t1 - t2)(1 - a·b) + t3((c·a) - (c·b)) = 0
                # This is getting complicated. Let me just compute it numerically.
                
                # Simpler: the circumcenter of a spherical triangle is the pole of the great circle
                # through the three points... no, that's the plane.
                
                # Actually, for three points on a sphere, the circumcenter is the center of the unique
                # circle passing through them. For a spherical triangle, if the triangle is small enough,
                # the circumcenter is approximately the centroid projected to the sphere, but that's not exact.
                
                # Let me use the linear algebra approach:
                # C = M * w where M = [a, b, c] (3x3 matrix with a,b,c as columns)
                # C·a = cos_r, C·b = cos_r, C·c = cos_r
                # M^T * C = [cos_r, cos_r, cos_r]^T
                # C = (M^T)^{-1} * [cos_r, cos_r, cos_r]^T = (M^T)^{-1} * cos_r * [1,1,1]^T
                # So C is proportional to (M^T)^{-1} * [1,1,1]^T
                # And cos_r = C·a = [(M^T)^{-1} * [1,1,1]^T] · a * scale... 
                # Actually C = s * (M^T)^{-1} * [1,1,1]^T for some scale s
                # Then C·a = s * ((M^T)^{-1} * [1,1,1]^T)·a = cos_r
                # So s = cos_r / (((M^T)^{-1} * [1,1,1]^T)·a)
                # But we also need C to be unit: ||C|| = 1
                # This is getting messy. Let me just use a simpler approach.
                
                # For the covering radius of a face, it's the angular distance from the circumcenter
                # to the farthest vertex. For a spherical triangle, the circumradius r satisfies:
                # sin(r) = (a×b·(b×c·(c×a))) / (2 * |a×b| * |b×c| * |c×a|) ... no that's not right either.
                
                # Let me just compute it directly:
                # The circumcenter C of the spherical triangle is the point on the sphere equidistant
                # from a, b, c. We can find it by:
                # C = normalize(a + b + c) is the centroid, not circumcenter.
                # 
                # The correct formula: the circumcenter is in the direction:
                # (b×c) × (c×a) + (c×a) × (a×b) + (a×b) × (b×c)
                # Wait no, that's the incenter.
                #
                # Let me just solve the system numerically:
                # C·a = C·b = C·c, ||C|| = 1
                # C·a = C·b => C·(a-b) = 0
                # C·b = C·c => C·(b-c) = 0
                # So C is perpendicular to (a-b) and (b-c), meaning C is parallel to (a-b)×(b-c)
                # But (a-b)×(b-c) is in the plane of a,b,c, which is correct for the circumcenter
                # of the spherical triangle (which lies in the plane of the triangle).
                
                ab = a[idx] - b[idx]
                bc = b[idx] - c[idx]
                cross = np.cross(ab, bc)
                norm_cross = np.linalg.norm(cross)
                if norm_cross < 1e-10:
                    # Degenerate triangle, use largest edge
                    d_ab = np.arccos(np.clip(np.dot(ai, bi), -1, 1))
                    d_bc = np.arccos(np.clip(np.dot(bi, ci), -1, 1))
                    d_ca = np.arccos(np.clip(np.dot(ci, ai), -1, 1))
                    radii[idx] = max(d_ab, d_bc, d_ca)
                else:
                    C = cross / norm_cross
                    # Check which side: C should be on the same side as the triangle
                    # The circumradius is arccos(C·a)
                    cos_r = np.dot(C, ai)
                    cos_r = np.clip(cos_r, -1, 1)
                    r = np.arccos(cos_r)
                    # If r > pi/2, the circumcenter is on the wrong side
                    if r > np.pi / 2:
                        r = np.pi - r
                    radii[idx] = r

            return float(np.max(radii))
        except Exception:
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def soft_min_covering_grad(pts, beta=30.0):
        """
        Compute soft-min covering radius and its gradient w.r.t. pts.
        Returns (objective, gradient) where gradient shape is (N, 3).
        """
        dot = np.dot(grid_pts, pts.T)  # (n_grid, N)
        dot_clipped = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot_clipped)  # (n_grid, N)

        # Soft-min over points
        neg_dists = -ang_dists
        max_neg = np.max(neg_dists, axis=1, keepdims=True)
        exp_neg = np.exp(beta * (neg_dists - max_neg))
        sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)
        w_min = exp_neg / sum_exp_neg  # (n_grid, N)
        soft_min_vals = -(max_neg[:, 0] + np.log(sum_exp_neg[:, 0]) / beta)

        # Soft-max over grid points
        max_sm = np.max(soft_min_vals)
        exp_sm = np.exp(beta * (soft_min_vals - max_sm))
        sum_exp_sm = np.sum(exp_sm)
        w_grid = exp_sm / sum_exp_sm  # (n_grid,)
        soft_max_val = max_sm + np.log(sum_exp_sm) / beta

        # Gradient computation
        sqrt_term = np.sqrt(np.maximum(1.0 - dot_clipped ** 2, 1e-15))  # (n_grid, N)
        weights = w_grid[:, np.newaxis] * w_min / sqrt_term  # (n_grid, N)
        grad = weights.T @ grid_pts  # (N, 3)

        return soft_max_val, grad

    def lloyd_iteration(points, n_iter=50):
        """
        Lloyd-like iteration: assign each grid point to nearest config point,
        then move each config point to the centroid of its assigned grid points.
        """
        for it in range(n_iter):
            if time.time() > deadline - 15.0:
                break
            
            # Assign each grid point to nearest config point
            dot = np.dot(grid_pts, points.T)  # (n_grid, N)
            nearest = np.argmax(dot, axis=1)  # (n_grid,)
            
            # Compute centroid for each config point
            new_points = np.zeros_like(points)
            for i in range(N):
                mask = nearest == i
                if np.sum(mask) > 0:
                    centroid = np.mean(grid_pts[mask], axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        new_points[i] = centroid / norm
                    else:
                        new_points[i] = points[i]
                else:
                    new_points[i] = points[i]
            
            points = new_points
        
        return points

    def repulsion_polish(points, alpha, n_iter=30):
        """Simple repulsion-based polish."""
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

    best_points = None
    best_rad = np.inf

    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() > deadline - 35.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            points = points + 0.1 * np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                points = best_points + 0.15 * np.random.randn(N, 3)
            else:
                points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Phase 1: Lloyd iteration for fast convergence
        if time.time() < deadline - 25.0:
            points = lloyd_iteration(points, n_iter=30)
            rad = compute_covering_radius(points)
            if rad < best_rad:
                best_rad = rad
                best_points = points.copy()

        # Phase 2: L-BFGS-B optimization with soft-min covering objective
        if time.time() < deadline - 15.0 and best_points is not None:
            x0 = best_points.flatten().copy()

            def objective_and_grad(x):
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-15] = 1.0
                pts = pts / norms
                obj, grad = soft_min_covering_grad(pts, beta=30.0)
                dot_g_p = np.sum(grad * pts, axis=1, keepdims=True)
                grad_tangent = grad - dot_g_p * pts
                return obj, grad_tangent.flatten()

            try:
                result = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                opt_pts = result.x.reshape(N, 3)
                opt_pts = opt_pts / np.linalg.norm(opt_pts, axis=1, keepdims=True)
                rad_opt = compute_covering_radius(opt_pts)
                if rad_opt < best_rad:
                    best_rad = rad_opt
                    best_points = opt_pts.copy()
            except Exception:
                pass

            # Phase 3: Second L-BFGS-B pass with higher beta
            if time.time() < deadline - 10.0 and best_points is not None:
                x0 = best_points.flatten().copy()

                def objective_and_grad2(x):
                    pts = x.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms[norms < 1e-15] = 1.0
                    pts = pts / norms
                    obj, grad = soft_min_covering_grad(pts, beta=50.0)
                    dot_g_p = np.sum(grad * pts, axis=1, keepdims=True)
                    grad_tangent = grad - dot_g_p * pts
                    return obj, grad_tangent.flatten()

                try:
                    result2 = minimize(
                        objective_and_grad2,
                        x0,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 40, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    opt_pts2 = result2.x.reshape(N, 3)
                    opt_pts2 = opt_pts2 / np.linalg.norm(opt_pts2, axis=1, keepdims=True)
                    rad_opt2 = compute_covering_radius(opt_pts2)
                    if rad_opt2 < best_rad:
                        best_rad = rad_opt2
                        best_points = opt_pts2.copy()
                except Exception:
                    pass

            # Phase 4: Final repulsion polish
            if time.time() < deadline - 5.0 and best_points is not None:
                polished = repulsion_polish(best_points.copy(), 0.003, 40)
                rad_polished = compute_covering_radius(polished)
                if rad_polished < best_rad:
                    best_rad = rad_polished
                    best_points = polished.copy()

    # Final refinement with L-BFGS-B
    if time.time() < deadline - 5.0 and best_points is not None:
        x0 = best_points.flatten().copy()

        def final_obj_grad(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            obj, grad = soft_min_covering_grad(pts, beta=80.0)
            dot_g_p = np.sum(grad * pts, axis=1, keepdims=True)
            grad_tangent = grad - dot_g_p * pts
            return obj, grad_tangent.flatten()

        try:
            result_final = minimize(
                final_obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 30, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            final_pts = result_final.x.reshape(N, 3)
            final_pts = final_pts / np.linalg.norm(final_pts, axis=1, keepdims=True)
            rad_final = compute_covering_radius(final_pts)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
