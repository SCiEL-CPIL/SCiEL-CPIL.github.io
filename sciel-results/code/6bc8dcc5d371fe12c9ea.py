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
GRID_FINE = _make_fib_grid(4000) # Reduced from 8000 to allow more iterations

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
            
            radii = np.zeros(len(faces))
            
            for i in range(len(faces)):
                a_i = a[i]
                b_i = b[i]
                c_i = c[i]
                
                n1 = np.cross(a_i, b_i)
                n2 = np.cross(b_i, c_i)
                u_dir = np.cross(n1, n2)
                
                if np.linalg.norm(u_dir) < 1e-10:
                    d_ab = np.arccos(np.clip(np.dot(a_i, b_i), -1, 1))
                    d_bc = np.arccos(np.clip(np.dot(b_i, c_i), -1, 1))
                    d_ca = np.arccos(np.clip(np.dot(c_i, a_i), -1, 1))
                    radii[i] = np.max([d_ab, d_bc, d_ca])
                    continue
                    
                u_dir /= np.linalg.norm(u_dir)
                centroid = (a_i + b_i + c_i) / 3.0
                if np.dot(u_dir, centroid) < 0:
                    u_dir = -u_dir
                
                cos_r = np.dot(u_dir, a_i)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                radii[i] = np.arccos(cos_r)
                
            return float(np.max(radii))
        except Exception:
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def smooth_objective_and_grad(x_flat, grid_pts, beta, alpha):
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)
        
        dot = np.dot(grid_pts, pts_norm.T)
        dot = np.clip(dot, -1.0 + 1e-12, 1.0 - 1e-12)
        ang_dists = np.arccos(dot)

        # Softmin
        max_d = np.max(ang_dists, axis=1, keepdims=True)
        exp_neg_beta_d = np.exp(-beta * (ang_dists - max_d))
        sum_exp_neg = np.sum(exp_neg_beta_d, axis=1)
        soft_min_dists = max_d - 1.0 / beta * np.log(sum_exp_neg)
        
        w_jk = exp_neg_beta_d / sum_exp_neg[:, np.newaxis]

        # Softmax
        max_s = np.max(soft_min_dists)
        exp_alpha_s = np.exp(alpha * (soft_min_dists - max_s))
        sum_exp_pos = np.sum(exp_alpha_s)
        val = max_s + 1.0 / alpha * np.log(sum_exp_pos)
        
        v_j = exp_alpha_s / sum_exp_pos

        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10))
        factor = (v_j[:, np.newaxis] * w_jk) / denom
        
        grad_pts_norm = -(factor.T @ grid_pts)
        
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200, beta=10.0, alpha=10.0):
        def objective(x):
            val, _ = smooth_objective_and_grad(x, grid_pts, beta, alpha)
            return val
        
        def gradient(x):
            _, g = smooth_objective_and_grad(x, grid_pts, beta, alpha)
            return g
        
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

    def polish_exact(points, iterations=20):
        """
        Local search to reduce the max circumradius of hull facets.
        Moves vertices of large facets towards their circumcenters.
        """
        pts = points.copy()
        for _ in range(iterations):
            if time.time() > deadline - 2.0:
                break
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                radii = np.zeros(len(faces))
                centers = np.zeros((len(faces), 3))
                
                for i in range(len(faces)):
                    a_i = a[i]
                    b_i = b[i]
                    c_i = c[i]
                    
                    n1 = np.cross(a_i, b_i)
                    n2 = np.cross(b_i, c_i)
                    u_dir = np.cross(n1, n2)
                    
                    if np.linalg.norm(u_dir) < 1e-10:
                        radii[i] = 0.0
                        centers[i] = (a_i + b_i + c_i) / 3.0
                        continue
                        
                    u_dir /= np.linalg.norm(u_dir)
                    centroid = (a_i + b_i + c_i) / 3.0
                    if np.dot(u_dir, centroid) < 0:
                        u_dir = -u_dir
                    
                    cos_r = np.dot(u_dir, a_i)
                    cos_r = np.clip(cos_r, -1.0, 1.0)
                    radii[i] = np.arccos(cos_r)
                    centers[i] = u_dir
                
                max_rad = np.max(radii)
                if max_rad < 1e-6:
                    break
                    
                # Identify facets with large radii
                # Move vertices towards centers
                # Use a small step size
                step = 0.05
                
                # Accumulate moves for each point
                moves = np.zeros_like(pts)
                counts = np.zeros(N)
                
                # Threshold for "large" facets
                threshold = max_rad - 1e-4
                
                for i in range(len(faces)):
                    if radii[i] > threshold:
                        idxs = faces[i]
                        verts = pts[idxs]
                        center = centers[i]
                        
                        # Move each vertex towards center
                        for k in range(3):
                            v = verts[k]
                            # Project move onto tangent space
                            move_dir = center - v
                            move_dir -= np.dot(move_dir, v) * v
                            move_dir /= np.linalg.norm(move_dir)
                            moves[idxs[k]] += move_dir * step
                            counts[idxs[k]] += 1
                
                # Apply moves
                for i in range(N):
                    if counts[i] > 0:
                        moves[i] /= counts[i]
                        pts[i] += moves[i]
                        pts[i] /= np.linalg.norm(pts[i])
                        
            except Exception:
                break
                
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Optimization from multiple starts
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
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
        # Use adaptive beta/alpha
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            # Start smooth, then sharper
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=50, beta=2.0, alpha=2.0)
            if time.time() < deadline - 10.0:
                points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=50, beta=5.0, alpha=5.0)
            
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Refinement with fine grid
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(3):
            if time.time() > deadline - 15.0:
                break
            
            candidate = best_points.copy()
            
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

    # Phase 3: Final Polish with Exact Local Search
    if best_points is not None and time.time() < deadline - 5.0:
        final = polish_exact(best_points, iterations=50)
        rad_final = compute_covering_radius_exact(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
