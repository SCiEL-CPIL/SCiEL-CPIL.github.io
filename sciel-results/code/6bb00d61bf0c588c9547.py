# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def fibonacci_sphere(n_pts, R=1.0):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts * R

    def lbfgs_optimize_min_dist(points, max_iter=500, time_limit=120):
        """
        Use L-BFGS-B to maximize the minimum pairwise distance on the unit sphere.
        We parameterize each point by its x,y coordinates and derive z from the sphere constraint.
        Actually, let's use a simpler approach: optimize in 3D with a sphere constraint penalty,
        or better: use spherical coordinates.
        
        Let's use the approach where we optimize the flattened 3D coordinates but project
        to sphere after each evaluation. L-BFGS-B doesn't handle constraints well, so let's
        use a penalty method or switch to a different parameterization.
        
        Alternative: Use scipy.optimize with the sphere constraint handled by projecting.
        We'll minimize -d_min with a penalty for deviation from unit sphere.
        """
        def objective_and_grad(flat):
            pts = flat.reshape(n, 3)
            # Project to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1e-12, norms)
            pts_s = pts / norms
            
            # Compute all pairwise distances
            diff = pts_s[:, np.newaxis, :] - pts_s[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(dist_sq)
            np.fill_diagonal(dist_matrix, np.inf)
            
            d_min = np.min(dist_matrix)
            
            # Penalty for not being on unit sphere
            sphere_penalty = np.sum((norms - 1.0) ** 2)
            
            # We want to maximize d_min, so minimize -d_min + penalty
            val = -d_min + 10.0 * sphere_penalty
            
            # Gradient
            # Gradient of d_min w.r.t. pts_s
            grad_dmin = np.zeros_like(pts_s)
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            i, j = min_idx
            if i != j:
                vec = pts_s[i] - pts_s[j]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad_dmin[i] += unit
                    grad_dmin[j] -= unit
            
            # Chain rule: pts_s = pts / norms
            # This is complex; let's use finite differences or a simpler approach
            # For now, let's just return the value and let L-BFGS use numerical gradients
            return val
        
        def objective(flat):
            return objective_and_grad(flat)
        
        initial_flat = project_to_sphere(points.copy()).flatten()
        
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 5000, 'maxls': 20}
        )
        
        opt_pts = project_to_sphere(result.x.reshape(n, 3))
        return opt_pts

    def hybrid_optimize(points, time_budget=100):
        """
        Hybrid optimization: use L-BFGS-B for the main optimization.
        """
        if time.time() - start_time > timeout - 20:
            return project_to_sphere(points.copy())
        
        # First, do a quick local optimization using L-BFGS-B
        t_start = time.time()
        max_iter = 200
        try:
            optimized = lbfgs_optimize_min_dist(points, max_iter=max_iter, time_limit=time_budget)
        except Exception:
            optimized = project_to_sphere(points.copy())
        
        elapsed = time.time() - t_start
        if elapsed < time_budget * 0.7:
            # Do a second round with more iterations
            try:
                optimized2 = lbfgs_optimize_min_dist(optimized, max_iter=300, time_limit=time_budget - elapsed)
                if calculate_score(optimized2) > calculate_score(optimized):
                    optimized = optimized2
            except Exception:
                pass
        
        return optimized

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []

    # 1. Fibonacci sphere
    inits.append(fibonacci_sphere(n))

    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    # 3. Spherical cap variations - start from fibonacci with perturbation
    for pert in [0.1, 0.2, 0.3, 0.5]:
        np.random.seed(999 + int(pert * 100))
        inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * pert)

    # 4. More structured inits: icosahedron-based
    # Regular icosahedron vertices (12 vertices) + 2 more
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 more points (e.g., along z-axis with perturbation)
    extra = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    full_ico = np.vstack([ico, extra])
    inits.append(full_ico)

    # 5. Octahedron-based + more
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ])
    # Add 8 more points at midpoints of edges
    edge_mids = []
    for i in range(len(octa)):
        for j in range(i+1, len(octa)):
            mid = (octa[i] + octa[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_mids.append(mid)
    # Take 8 of them
    edge_mids = np.array(edge_mids[:8])
    inits.append(np.vstack([octa, edge_mids]))

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break

        pts = project_to_sphere(init.copy())
        
        # Optimize with time budget
        remaining = timeout - 30 - (time.time() - start_time)
        if remaining < 10:
            break
            
        time_per_init = min(remaining / max(1, len(inits) - init_idx), 120)
        optimized = hybrid_optimize(pts, time_budget=time_per_init)
        
        s = calculate_score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
