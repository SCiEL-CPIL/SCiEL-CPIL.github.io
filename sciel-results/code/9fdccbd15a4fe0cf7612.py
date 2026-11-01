import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin before 120s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
        
        # Vectorized circumcenter computation
        a = p[facets[:, 0]]  # (F, 3)
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        
        v1 = a - b  # (F, 3)
        v2 = b - c
        n = np.cross(v1, v2)  # (F, 3)
        norm_n = np.linalg.norm(n, axis=1)  # (F,)
        
        # Handle degenerate facets
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def soft_max_objective(params, beta=200.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        As beta -> inf, this approaches the true max.
        """
        X = params.reshape(N, 3)
        X = normalize(X)
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        
        # Soft-max: (1/beta) * log(sum(exp(beta * radii)))
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def repulsion_relax(pts, n_iters=500, alpha_start=0.05):
        """
        Repulsion relaxation with adaptive step size (backtracking line search).
        Minimize sum 1/||p_i - p_j||^2 via gradient descent on sphere.
        This ensures points are well-spread before structural optimization.
        """
        pts = pts.copy()
        alpha = alpha_start
        
        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)
        
        current_energy = repulsion_energy(pts)
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Calculate pairwise differences
            diff = pts[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            
            # Avoid division by zero
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
            
            # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Project onto tangent plane: F_tan = F - (F . p) p
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
            # Adaptive step size with backtracking
            step = alpha
            improved = False
            for _ in range(10):
                new_pts = pts + step * forces_tan
                new_pts = normalize(new_pts)
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                alpha *= 0.8
            else:
                alpha *= 1.01  # Slightly increase if successful
            
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)
        
        return pts

    def lloyd_iteration(pts, n_iters=80):
        """
        Improved Lloyd's algorithm: move each point toward the circumcenter
        of its worst facet, with adaptive step size.
        This is a structural move that directly reduces the covering radius.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            # For each point, find the maximum radius facet it participates in
            # and move toward that facet's circumcenter
            updates = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for f in range(len(facets)):
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                # Weight by how much this facet contributes to the max
                # Using squared weight emphasizes the worst facets more
                weight = (rad / max_rad) ** 2
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight
            
            # Apply updates where counts > 0
            mask = counts > 1e-10
            if np.any(mask):
                # Adaptive step size based on radius spread
                step_size = 0.05 * (max_rad / (np.mean(radii) + 1e-15))
                step_size = min(step_size, 0.1)  # Cap step size
                pts[mask] += step_size * updates[mask] / counts[mask, np.newaxis]
            
            # Normalize
            pts = normalize(pts)
        
        return pts

    def optimize_from_start(pts_init, obj_func, maxiter=100):
        """Run L-BFGS-B optimization from a starting configuration."""
        best_pts_local = pts_init.copy()
        best_radius_local = get_max_radius(pts_init)
        
        # Normalize initial points
        pts_init = normalize(pts_init)
        
        try:
            res = minimize(obj_func, pts_init.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})
            
            if res.fun < obj_func(pts_init.flatten()):
                new_pts = res.x.reshape(N, 3)
                new_pts = normalize(new_pts)
                
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_pts_local = new_pts.copy()
        except Exception:
            pass
        
        return best_pts_local, best_radius_local

    # Generate initial configurations
    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase)
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def greedy_init(seed_val):
        rng = np.random.RandomState(seed_val)
        pts = [rng.randn(3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.randn(2000, 3)
            candidates = normalize(candidates)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)

    initial_configs = [
        fibonacci_initial(0.0),
        fibonacci_initial(0.37),
        fibonacci_initial(0.618),
        greedy_init(101),
        greedy_init(202)
    ]
    
    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)

    # Phase 1: Repulsion Relaxation to spread points
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 40.0:
            break
            
        relaxed_pts = repulsion_relax(init_pts, n_iters=300)
        
        # Phase 2: Lloyd's Algorithm for structural refinement
        if time.time() > deadline - 35.0:
            break
            
        lloyd_pts = lloyd_iteration(relaxed_pts, n_iters=60)
        
        # Check improvement from Lloyd's
        lloyd_radius = get_max_radius(lloyd_pts)
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best_pts = lloyd_pts.copy()

    # Phase 3: L-BFGS Soft-Max Polish from best Lloyd result
    if time.time() < deadline - 10.0:
        # Use a high beta for sharp minimax approximation
        def obj_polish(params):
            return soft_max_objective(params, beta=300.0)
        
        polished_pts, polished_radius = optimize_from_start(best_pts, obj_polish, maxiter=150)
        
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_pts = polished_pts.copy()

    # Phase 4: Final fine-tuning with lower beta for robustness
    if time.time() < deadline - 2.0:
        def obj_fine(params):
            return soft_max_objective(params, beta=100.0)
        
        fine_pts, fine_radius = optimize_from_start(best_pts, obj_fine, maxiter=100)
        
        if fine_radius < best_radius:
            best_radius = fine_radius
            best_pts = fine_pts.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()
    
    best_pts = normalize(best_pts)
    
    return best_pts
