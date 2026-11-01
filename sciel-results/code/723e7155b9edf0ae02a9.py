"""Program for covering the sphere with 130 equal caps using soft-attention gradient optimization
with exact hull-based local refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s hard limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Dense grid for accurate gradient and evaluation
    grid_main = generate_grid(120, 240)

    def lloyd_covering_step(pts, grid, n_iters=50, alpha=0.5):
        current = pts.copy()
        for it in range(n_iters):
            if time.time() > deadline - 15.0:
                break
            dots = np.dot(grid, current.T)
            nearest = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current)
            for i in range(N):
                mask = (nearest == i)
                if not np.any(mask):
                    new_pts[i] = current[i]
                    continue
                cell_pts = grid[mask]
                d = current[i] @ cell_pts.T
                d = np.clip(d, -1.0, 1.0)
                ang = np.arccos(d)
                temp = 3.0
                w = np.exp(ang * temp)
                w /= np.sum(w)
                new_c = np.sum(cell_pts * w[:, None], axis=0)
                nrm = np.linalg.norm(new_c)
                if nrm < 1e-10:
                    new_pts[i] = current[i]
                else:
                    new_pts[i] = new_c / nrm
            current = (1 - alpha) * current + alpha * new_pts
            current /= np.linalg.norm(current, axis=1, keepdims=True)
        return current

    def covering_radius(pts, grid=None):
        if grid is None:
            grid = grid_main
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def exact_covering_radius(pts):
        """Compute exact covering radius from convex hull."""
        try:
            hull = ConvexHull(pts)
        except Exception:
            return covering_radius(pts)
        
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            max_radius = max(max_radius, facet_radius)
        return max_radius

    def soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
        """
        Vectorized soft covering radius and its gradient using smooth soft-min over all points.
        """
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        
        neg_scaled = -angles / sigma  # (M, N)
        max_neg = np.max(neg_scaled, axis=1, keepdims=True)
        exp_neg = np.exp(neg_scaled - max_neg)
        weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
        
        soft_min_vals = np.sum(weights * angles, axis=1)
        
        scaled_max = soft_min_vals / sigma
        max_scaled = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_scaled)
        max_weights = exp_max / np.sum(exp_max)
        soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
        
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))
        coeff = max_weights[:, np.newaxis] * weights
        alpha = coeff / denom
        term1 = -(alpha.T @ grid_points)
        beta = alpha * dots
        beta_sum = np.sum(beta, axis=0)
        term2 = beta_sum[:, np.newaxis] * points
        grad = term1 + term2
        
        return soft_max_val, grad

    def optimize_with_annealing(pts, sigmas, max_iter_per_stage=20):
        """Multi-stage gradient optimization with sigma annealing."""
        if time.time() > deadline - 15.0:
            return pts
        
        current = pts.copy()
        
        for sigma in sigmas:
            if time.time() > deadline - 10.0:
                break
            
            def objective_and_grad(x_flat, sigma=sigma):
                pts_local = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts_local = pts_local / norms
                val, grad = soft_covering_radius_and_grad(pts_local, grid_main, sigma=sigma)
                return val, grad.flatten()
            
            x0 = current.flatten()
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_per_stage, 'ftol': 1e-10, 'gtol': 1e-5}
                )
                if res.fun < objective_and_grad(x0)[0]:
                    p = res.x.reshape(N, 3)
                    p /= np.linalg.norm(p, axis=1, keepdims=True)
                    current = p
            except Exception:
                pass
        
        return current

    def exact_local_optimize(pts, max_iters=30, step_size=0.02):
        """
        Exact hull-based local optimization: for each point, identify its worst facet
        and move it toward the circumcenter of that facet to reduce the facet radius.
        """
        pts = pts.copy()
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Compute all facet data
            facet_data = []
            for simplex in hull.simplices:
                verts = pts[simplex]
                a, b, c = verts[0], verts[1], verts[2]
                normal = np.cross(b - a, c - a)
                norm_val = np.linalg.norm(normal)
                if norm_val < 1e-12:
                    continue
                normal = normal / norm_val
                centroid = np.mean(verts, axis=0)
                if np.dot(normal, centroid) < 0:
                    normal = -normal
                dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
                angles = np.arccos(dots)
                facet_radius = np.max(angles)
                facet_data.append((simplex, normal, facet_radius))
            
            if not facet_data:
                break
            
            # Find the worst facet
            worst_idx = np.argmax([fd[2] for fd in facet_data])
            worst_simplex, worst_normal, worst_radius = facet_data[worst_idx]
            
            # For each point in the worst facet, compute a move toward the circumcenter
            # The circumcenter of a spherical triangle is along the normal direction
            # We want to reduce the max angle from vertices to circumcenter
            # Moving a vertex toward the circumcenter reduces its angle to the circumcenter
            
            improved = False
            for j in worst_simplex:
                if time.time() > deadline - 10.0:
                    break
                
                p_j = pts[j]
                # Direction toward circumcenter (which is along the normal)
                # But we need to move in the tangent plane
                # The circumcenter is at normal direction; moving p_j toward normal
                # reduces the angle between p_j and normal
                
                # Project the desired move onto the tangent plane at p_j
                # Move direction: from p_j toward normal
                move_dir = worst_normal - np.dot(p_j, worst_normal) * p_j
                move_norm = np.linalg.norm(move_dir)
                if move_norm < 1e-10:
                    continue
                move_dir /= move_norm
                
                # Small step
                new_p = p_j + step_size * move_dir
                new_p /= np.linalg.norm(new_p)
                
                # Check if this improves the worst facet radius
                # Compute new facet radius if we move point j
                new_pts = pts.copy()
                new_pts[j] = new_p
                
                # Recompute the worst facet radius
                try:
                    new_hull = ConvexHull(new_pts)
                except Exception:
                    continue
                
                new_max_radius = 0.0
                for simplex in new_hull.simplices:
                    verts = new_pts[simplex]
                    a, b, c = verts[0], verts[1], verts[2]
                    normal = np.cross(b - a, c - a)
                    norm_val = np.linalg.norm(normal)
                    if norm_val < 1e-12:
                        continue
                    normal = normal / norm_val
                    centroid = np.mean(verts, axis=0)
                    if np.dot(normal, centroid) < 0:
                        normal = -normal
                    dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
                    angles = np.arccos(dots)
                    new_max_radius = max(new_max_radius, np.max(angles))
                
                if new_max_radius < worst_radius - 1e-10:
                    pts[j] = new_p
                    improved = True
            
            if not improved:
                # Try a smaller step
                step_size *= 0.7
                if step_size < 1e-5:
                    break
        
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 50.0:
            break
        
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = max(0.01, 0.04 - 0.005 * (r - 4))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)
        
        # Phase 1: Lloyd iterations for coarse optimization
        cand = lloyd_covering_step(pts, grid_main, n_iters=25, alpha=0.5)
        
        # Phase 2: Multi-stage soft-attention gradient optimization
        sigmas = [0.15, 0.08, 0.04, 0.02]
        cand = optimize_with_annealing(cand, sigmas, max_iter_per_stage=12)
        
        # Phase 3: Exact hull-based local optimization
        if time.time() < deadline - 25.0:
            cand = exact_local_optimize(cand, max_iters=20, step_size=0.02)
        
        # Phase 4: Final fine-tuning with very small sigma
        if time.time() < deadline - 15.0:
            cand = optimize_with_annealing(cand, [0.01, 0.005], max_iter_per_stage=8)
        
        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    # Final refinement on best solution
    if best_pts is not None and time.time() < deadline - 15.0:
        fp = lloyd_covering_step(best_pts, grid_main, n_iters=10, alpha=0.3)
        fp = optimize_with_annealing(fp, [0.01, 0.005], max_iter_per_stage=8)
        fp = exact_local_optimize(fp, max_iters=15, step_size=0.01)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
