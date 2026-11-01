"""Program for covering the sphere with 130 equal caps using smooth grid-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective
# We use a Fibonacci lattice for uniform distribution
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# Grid sizes: 
# COARSE for fast initial optimization
# FINE for final polish
GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety (150s limit)

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """
        Compute covering radius exactly as max circumradius over convex hull faces.
        Returns radius in radians.
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Circumradius of a spherical triangle (a,b,c)
            # The circumcenter is the normalized cross product sum? No.
            # The circumcenter of a triangle on the sphere is the intersection of the great circles.
            # It is proportional to a x (b x c) + b x (c x a) + c x (a x b) ?
            # Actually, the vector n = (a x b) + (b x c) + (c x a) points towards the circumcenter?
            # Let's check. For an equilateral triangle, yes.
            # A more robust formula:
            # The circumcenter u satisfies u . a = u . b = u . c = cos(R).
            # u is in the span of a, b, c.
            # We can solve the linear system:
            # [a^T; b^T; c^T] u = [cos R; cos R; cos R]
            # This is 3x3.
            
            radii = np.zeros(len(faces))
            
            # Vectorized solution for circumradius
            # u = (a x (b x c) + b x (c x a) + c x (a x b)) / |...| ?
            # Let's use the property that the circumcenter is the normalized sum of the vertices weighted by the opposite side lengths? No.
            
            # Simple and robust: Solve linear system for each face.
            # But vectorized:
            # Let M be matrix with rows a, b, c.
            # u = M^T (M M^T)^{-1} [1, 1, 1]^T ? No, u . a = cos R.
            # Let A = [a, b, c] (3x3).
            # We want u such that A^T u = [cos R, cos R, cos R]^T.
            # u = A^{-T} [cos R, cos R, cos R]^T.
            # Also |u| = 1.
            # u = A^{-T} v, where v = c_r * [1, 1, 1]^T.
            # |A^{-T} v| = 1 => c_r * |A^{-T} [1,1,1]| = 1.
            # c_r = 1 / |A^{-T} [1,1,1]|.
            # cos R = u . a = (A^{-T} v) . a = v . (A^{-1} a) = c_r * [1,1,1] . (A^{-1} a).
            
            # This is getting complicated to vectorize with inv.
            # Alternative:
            # The circumradius R is given by:
            # sin(R) = |a x b x c| / (|a x b| |b x c| |c x a|) ? No.
            # Area of spherical triangle: E = A + B + C - pi.
            # sin(E/2) = ...
            
            # Let's stick to the previous method which was likely correct:
            # The circumcenter direction is n = (a x b) + (b x c) + (c x a) ?
            # Let's test with a right triangle at the pole.
            # a=(0,0,1), b=(1,0,0), c=(0,1,0).
            # a x b = (0,1,0). b x c = (1,0,0). c x a = (-1,0,0)? No c x a = (0,1,0) x (0,0,1) = (1,0,0).
            # Sum = (0,1,0) + (1,0,0) + (1,0,0) = (2,1,0).
            # Normalize: (2,1,0)/sqrt(5).
            # cos R = u . a = 0. R = pi/2.
            # Actual circumcenter of (0,0,1), (1,0,0), (0,1,0) is (1,1,1)/sqrt(3).
            # cos R = 1/sqrt(3). R = arccos(1/sqrt(3)) ~ 54.7 deg.
            # So the sum of cross products is NOT the circumcenter.
            
            # Correct formula for circumcenter of spherical triangle:
            # u is proportional to a/sin(A) + b/sin(B) + c/sin(C) ? No.
            
            # Let's just solve the 3x3 system. It's fast enough for ~200 faces.
            for i in range(len(faces)):
                a_i = a[i]
                b_i = b[i]
                c_i = c[i]
                
                # Matrix A has columns a, b, c? No, rows.
                # We want u . a = cos R, u . b = cos R, u . c = cos R.
                # Let u = x a + y b + z c.
                # This is a linear system.
                # Simpler: u is in the span of a, b, c.
                # u = (a x (b x c) + b x (c x a) + c x (a x b)) is the normal to the plane? No.
                
                # Use the formula:
                # u = (a * sin(B) * sin(C) + b * sin(C) * sin(A) + c * sin(A) * sin(B)) ?
                # No, let's use the determinant formula.
                # |u| = 1.
                # u . a = u . b = u . c = cos R.
                
                # Let's use the property that the circumcenter is the intersection of the great circles.
                # Normal to circle through a,b is a x b.
                # Normal to circle through b,c is b x c.
                # u is perpendicular to both? No, u is in the plane of the great circles?
                # The great circle through a and b is the plane spanned by a and b. Normal is a x b.
                # So u . (a x b) = 0.
                # u . (b x c) = 0.
                # u . (c x a) = 0.
                # This implies u is parallel to (a x b) x (b x c).
                
                n1 = np.cross(a_i, b_i)
                n2 = np.cross(b_i, c_i)
                u_dir = np.cross(n1, n2)
                
                if np.linalg.norm(u_dir) < 1e-10:
                    # Degenerate, use max pair distance
                    d_ab = np.arccos(np.clip(np.dot(a_i, b_i), -1, 1))
                    d_bc = np.arccos(np.clip(np.dot(b_i, c_i), -1, 1))
                    d_ca = np.arccos(np.clip(np.dot(c_i, a_i), -1, 1))
                    radii[i] = np.max([d_ab, d_bc, d_ca])
                    continue
                    
                u_dir /= np.linalg.norm(u_dir)
                # Check orientation. u_dir might point to the other side.
                # We want the one closer to the triangle centroid?
                # Actually, there are two solutions. We want the one such that the triangle is "convex" in the right way.
                # Usually, the circumcenter is on the same side of the edges as the third vertex?
                # Let's check the dot product with the centroid (a+b+c)/3.
                centroid = (a_i + b_i + c_i) / 3.0
                if np.dot(u_dir, centroid) < 0:
                    u_dir = -u_dir
                
                cos_r = np.dot(u_dir, a_i)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                radii[i] = np.arccos(cos_r)
                
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def smooth_objective_and_grad(x_flat, grid_pts, beta=10.0, alpha=10.0):
        """
        Compute smooth approximation of covering radius and its gradient.
        Objective: Softmax( Softmin( distances ) )
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)
        
        # 1. Compute distances matrix D (M x N)
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0 + 1e-12, 1.0 - 1e-12)
        ang_dists = np.arccos(dot)  # M x N

        # 2. Softmin over points (axis=1)
        # s_j = -1/beta * log( sum_k exp(-beta * d_jk) )
        exp_neg_beta_d = np.exp(-beta * ang_dists)
        sum_exp_neg = np.sum(exp_neg_beta_d, axis=1)
        soft_min_dists = -1.0 / beta * np.log(sum_exp_neg)
        
        # Weights for softmin: w_jk = exp(-beta * d_jk) / sum_exp_neg
        w_jk = exp_neg_beta_d / sum_exp_neg[:, np.newaxis]  # M x N

        # 3. Softmax over grid points (axis=0)
        # R = 1/alpha * log( sum_j exp(alpha * s_j) )
        exp_alpha_s = np.exp(alpha * (soft_min_dists - np.max(soft_min_dists)))
        sum_exp_pos = np.sum(exp_alpha_s)
        val = np.max(soft_min_dists) + 1.0 / alpha * np.log(sum_exp_pos)
        
        # Weights for softmax: v_j = exp(alpha * s_j) / sum_exp_pos
        v_j = exp_alpha_s / sum_exp_pos  # M

        # 4. Gradient
        # dR/dp_k = sum_j v_j * (ds_j/dp_k)
        # ds_j/dp_k = sum_l w_jl * (d_d_jl/dp_k)
        # d_d_jl/dp_k = delta_lk * d_d_jl/dp_l
        # d_d_jl/dp_l = - grid_pts[j] / sqrt(1 - dot_jl^2)
        
        # So dR/dp_k = sum_j v_j * w_jk * ( - grid_pts[j] / sqrt(1 - dot_jk^2) )
        
        # denom = sqrt(1 - dot^2)
        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10))  # M x N
        
        # factor = v_j[:, np.newaxis] * w_jk / denom
        factor = (v_j[:, np.newaxis] * w_jk) / denom  # M x N
        
        # grad_pts_norm[k] = - sum_j factor[j, k] * grid_pts[j]
        grad_pts_norm = -(factor.T @ grid_pts)  # N x 3
        
        # Chain rule: pts = pts_norm * norm
        # We are optimizing x_flat, which are the unnormalized coordinates?
        # No, usually we optimize on the sphere directly or use a retraction.
        # L-BFGS-B works on Euclidean space.
        # If we treat x_flat as Euclidean coordinates, the constraint is ||x_i||=1.
        # The gradient on the sphere is the projection of the Euclidean gradient.
        # However, standard practice for L-BFGS-B on sphere is to optimize the unnormalized coordinates and normalize inside the objective, or use a penalty.
        # The current code normalizes inside `smooth_objective_and_grad`.
        # The gradient returned must be with respect to `x_flat`.
        
        # d(pts_norm)/d(x_flat) is not simply the identity.
        # pts_norm = x / ||x||.
        # d(pts_norm_i)/d(x_i) = (I - p_i p_i^T) / ||x_i||.
        
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # Project grad_pts_norm[i] to tangent space and scale
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200, beta=10.0, alpha=10.0):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = smooth_objective_and_grad(x, grid_pts, beta, alpha)
            return val
        
        def gradient(x):
            _, g = smooth_objective_and_grad(x, grid_pts, beta, alpha)
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

    best_points = None
    best_rad = np.inf

    # Phase 1: Optimization from multiple starts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
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
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Optimize with L-BFGS-B on coarse grid first for speed
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            # Use lower beta/alpha for coarser optimization if needed, but 10 is usually fine
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=100, beta=5.0, alpha=5.0)
            
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Refinement with fine grid
    if best_points is not None and time.time() < deadline - 15.0:
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Start from best points
            candidate = best_points.copy()
            
            # Small perturbation to escape local minima if stuck
            if trial > 0:
                rng = np.random.RandomState(500 + trial)
                candidate = candidate + 0.02 * rng.randn(N, 3)
                candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            # Optimize on fine grid
            time_left = deadline - time.time() - 5.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=100, beta=10.0, alpha=10.0)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Final Polish
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        # Very fine grid for final check? No, just optimize again on fine grid with tighter tol
        time_left = deadline - time.time() - 2.0
        if time_left > 2.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=50, beta=20.0, alpha=20.0)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
