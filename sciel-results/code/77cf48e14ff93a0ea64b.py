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
        This is smoother than sum(radii**power) for the minimax problem.
        """
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        
        # Soft-max: (1/beta) * log(sum(exp(beta * radii)))
        # For numerical stability, subtract max
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def power_objective(params, power=2.0):
        """Sum of radii^power as objective."""
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        return np.sum(radii**power)

    def optimize_from_start(pts_init, deadline_local, obj_func, maxiter=80):
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
    for phase in [0.37, 0.618, 0.25]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)
    
    # 3. Random points (seeded for reproducibility)
    np.random.seed(42)
    for seed_offset in range(3):
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
            candidates = np.random.randn(2000, 3)
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

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)
    
    # Phase 1: Quick optimization with soft-max (smooth, good for broad search)
    current_time = time.time()
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 15.0:
            break
        
        def obj_sm(params, _beta=100.0):
            return soft_max_objective(params, beta=_beta)
        
        pts_res, rad_res = optimize_from_start(init_pts, deadline - 15.0, obj_sm, maxiter=60)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
    
    # Phase 2: Refine with higher beta soft-max (closer to true max)
    if time.time() < deadline - 10.0:
        def obj_sm_high(params, _beta=500.0):
            return soft_max_objective(params, beta=_beta)
        
        pts_res, rad_res = optimize_from_start(best_pts, deadline - 10.0, obj_sm_high, maxiter=80)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
        
        # Try from second-best initial config too
        if time.time() < deadline - 8.0:
            # Find second best from phase 1
            second_best_idx = -1
            second_best_rad = float('inf')
            for idx, init_pts in enumerate(initial_configs):
                if time.time() > deadline - 8.0:
                    break
                rad_check = get_max_radius(init_pts)
                if rad_check < second_best_rad and rad_check > best_radius:
                    second_best_rad = rad_check
                    second_best_idx = idx
            
            if second_best_idx >= 0:
                pts_res2, rad_res2 = optimize_from_start(
                    initial_configs[second_best_idx], deadline - 8.0, obj_sm_high, maxiter=60)
                if rad_res2 < best_radius:
                    best_radius = rad_res2
                    best_pts = pts_res2.copy()
    
    # Phase 3: Final polish with power-based objective (tighter convergence)
    if time.time() < deadline - 3.0:
        def obj_power(params, _p=4.0):
            return power_objective(params, power=_p)
        
        try:
            res = minimize(obj_power, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 500})
            if res.fun < obj_power(best_pts.flatten()):
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
    
    # Phase 4: Ultra-tight final polish
    if time.time() < deadline - 1.0:
        def obj_final(params, _beta=1000.0):
            return soft_max_objective(params, beta=_beta)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300})
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
    
    return best_pts
