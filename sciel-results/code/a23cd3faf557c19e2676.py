import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin

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
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
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

    def repulsion_relax(pts, n_iters=800, alpha_start=0.08):
        """
        Repulsion relaxation with adaptive step size to spread points.
        Minimize sum 1/||p_i - p_j||^2 via gradient descent on sphere.
        """
        pts = pts.copy()
        alpha = alpha_start
        
        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return np.sum(inv_dist_sq)
        
        current_energy = repulsion_energy(pts)
        
        for it in range(n_iters):
            # Calculate pairwise differences
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
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
            best_alpha = alpha
            best_energy = current_energy
            for trial_alpha in [alpha, alpha * 0.5, alpha * 0.25, alpha * 0.1]:
                trial_pts = pts + trial_alpha * forces_tan
                norms = np.linalg.norm(trial_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                trial_pts = trial_pts / norms
                trial_energy = repulsion_energy(trial_pts)
                if trial_energy < best_energy:
                    best_energy = trial_energy
                    best_alpha = trial_alpha
                    pts = trial_pts
                    break
            
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            # Decay alpha
            alpha *= 0.995
            current_energy = best_energy
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def lloyd_iteration(pts, n_iters=50):
        """
        Improved Lloyd's algorithm: move ALL points toward their
        worst-facet circumcenters to equalize radii across the sphere.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            # For each point, accumulate displacement toward circumcenters
            # of facets it belongs to, weighted by facet radius
            disp = np.zeros((N, 3))
            
            for f_idx in range(len(facets)):
                facet = facets[f_idx]
                cc = circumcenters[f_idx]
                r = radii[f_idx]
                # Weight by how "bad" this facet is relative to max
                weight = (r / max_rad) ** 2
                for idx in facet:
                    disp[idx] += weight * (cc - pts[idx])
            
            # Normalize displacement
            disp_norms = np.linalg.norm(disp, axis=1, keepdims=True)
            disp_norms[disp_norms < 1e-12] = 1.0
            disp = disp / disp_norms
            
            # Small step toward displacement
            step_size = 0.02
            pts = pts + step_size * disp
            
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def facet_local_search(pts, n_iters=100):
        """
        Direct facet-targeting local search: repeatedly find the worst facet
        and perform a small local optimization to reduce its circumradius.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            # Find worst facet
            worst_idx = np.argmax(radii)
            worst_facet = facets[worst_idx]
            worst_cc = circumcenters[worst_idx]
            
            # Try small perturbations of the three vertices
            best_pts_local = pts.copy()
            best_rad_local = max_rad
            
            # Generate candidate moves for each vertex
            for v_idx in worst_facet:
                p_v = pts[v_idx].copy()
                # Try moving toward circumcenter
                for step in [0.005, 0.01, 0.02, 0.005]:
                    trial_pts = pts.copy()
                    trial_pts[v_idx] = p_v + step * (worst_cc - p_v)
                    norms = np.linalg.norm(trial_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    trial_pts = trial_pts / norms
                    
                    rad = get_max_radius(trial_pts)
                    if rad < best_rad_local:
                        best_rad_local = rad
                        best_pts_local = trial_pts.copy()
            
            pts = best_pts_local
            
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def optimize_from_start(pts_init, obj_func, maxiter=80):
        """Run L-BFGS-B optimization from a starting configuration."""
        best_pts_local = pts_init.copy()
        best_radius_local = get_max_radius(pts_init)
        
        # Normalize initial points
        norms = np.linalg.norm(pts_init, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts_init = pts_init / norms
        
        try:
            res = minimize(obj_func, pts_init.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})
            
            if res.fun < obj_func(pts_init.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_pts_local = new_pts.copy()
        except Exception:
            pass
            
        return best_pts_local, best_radius_local

    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initial_configs.append(fib_pts)
    
    # 2. Fibonacci with different phase offsets
    for phase in [0.37, 0.618, 0.25, 0.123, 0.789]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)
    
    # 3. Random points (seeded for reproducibility)
    for seed_offset in range(4):
        np.random.seed(42 + seed_offset * 100)
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)
        
    # 4. Greedy initialization - place points to maximize minimum distance
    def greedy_init(seed_val):
        np.random.seed(seed_val)
        pts = [np.random.randn(3)]
        norms = np.linalg.norm(pts[0])
        pts[0] = pts[0] / norms
        for _ in range(N - 1):
            candidates = np.random.randn(3000, 3)
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            candidates = candidates / norms
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)
    
    initial_configs.append(greedy_init(101))
    initial_configs.append(greedy_init(202))
    initial_configs.append(greedy_init(303))
    initial_configs.append(greedy_init(404))

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)
    
    # Phase 1: Repulsion relaxation to get good spread
    # Use more configs and more iterations for better initial spread
    for idx, init_pts in enumerate(initial_configs[:6]):
        if time.time() > deadline - 50.0:
            break
        
        relaxed = repulsion_relax(init_pts, n_iters=600, alpha_start=0.06)
        rad_relaxed = get_max_radius(relaxed)
        if rad_relaxed < best_radius:
            best_radius = rad_relaxed
            best_pts = relaxed.copy()
    
    # Phase 2: Quick optimization with soft-max (smooth, good for broad search)
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 35.0:
            break
        
        def obj_sm(params, _beta=100.0):
            return soft_max_objective(params, beta=_beta)
        
        pts_res, rad_res = optimize_from_start(init_pts, obj_sm, maxiter=60)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
    
    # Phase 3: Refine with higher beta soft-max (closer to true max)
    if time.time() < deadline - 25.0:
        def obj_sm_high(params, _beta=500.0):
            return soft_max_objective(params, beta=_beta)
        
        pts_res, rad_res = optimize_from_start(best_pts, obj_sm_high, maxiter=100)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
        
        # Try from other good initial configs
        if time.time() < deadline - 18.0:
            for idx, init_pts in enumerate(initial_configs[:4]):
                if time.time() > deadline - 18.0:
                    break
                rad_check = get_max_radius(init_pts)
                if rad_check < best_radius + 0.05:
                    pts_res2, rad_res2 = optimize_from_start(
                        init_pts, obj_sm_high, maxiter=70)
                    if rad_res2 < best_radius:
                        best_radius = rad_res2
                        best_pts = pts_res2.copy()
    
    # Phase 4: Lloyd's algorithm for local refinement (improved version)
    if time.time() < deadline - 12.0:
        lloyd_pts = lloyd_iteration(best_pts, n_iters=50)
        rad_lloyd = get_max_radius(lloyd_pts)
        if rad_lloyd < best_radius:
            best_radius = rad_lloyd
            best_pts = lloyd_pts.copy()
    
    # Phase 5: Facet-targeting local search
    if time.time() < deadline - 8.0:
        facet_pts = facet_local_search(best_pts, n_iters=80)
        rad_facet = get_max_radius(facet_pts)
        if rad_facet < best_radius:
            best_radius = rad_facet
            best_pts = facet_pts.copy()
    
    # Phase 6: Final polish with soft-max (tighter convergence)
    if time.time() < deadline - 5.0:
        def obj_final(params, _beta=1000.0):
            return soft_max_objective(params, beta=_beta)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 60, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300})
            if res.fun < obj_final(best_pts.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_pts = best_pts / norms
    
    return best_pts
