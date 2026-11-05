import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 8

    def compute_score(points):
        """
        Compute the exact score: product of distances over ordered pairs,
        after scaling so diameter = 2, then divide by n^n.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return 0.0
        
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return 0.0
        
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        
        log_prod = np.sum(np.log(scaled_dists))
        score = np.exp(log_prod - n * np.log(n))
        return score

    def polar_to_cartesian(polar):
        """Convert polar coordinates [r0, theta0, r1, theta1, ...] to Cartesian [x0, y0, x1, y1, ...]"""
        r = polar[0::2]
        theta = polar[1::2]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    def log_product_of_distances(polar):
        """
        Compute log of product of all pairwise distances (unordered pairs).
        This is what we want to maximize.
        """
        pts = polar_to_cartesian(polar)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(np.maximum(dists_sq, 1e-30))
        
        iu = np.triu_indices(n, k=1)
        off_dists = dists[iu]
        
        if np.any(off_dists < 1e-12):
            return -1e10  # Penalty for coincident points
        
        return np.sum(np.log(off_dists))

    def objective_polar(polar):
        """
        Objective function for optimization.
        We maximize log product of distances, so minimize the negative.
        """
        if time.time() > deadline - 5:
            return 1e10
        
        r = polar[0::2]
        theta = polar[1::2]
        
        # Check for invalid values
        if np.any(~np.isfinite(polar)):
            return 1e10
        if np.any(r < 0.01):
            return 1e10
        
        ld = log_product_of_distances(polar)
        
        if not np.isfinite(ld):
            return 1e10
        
        return -ld

    def full_objective_cartesian(flat_pts):
        """Objective in Cartesian coordinates for local refinement."""
        if time.time() > deadline - 5:
            return 1e10
        pts = flat_pts.reshape(8, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(np.maximum(dists_sq, 1e-30))
        
        iu = np.triu_indices(n, k=1)
        off_dists = dists[iu]
        
        if np.any(off_dists < 1e-12):
            return 1e10
        
        return -np.sum(np.log(off_dists))

    def cartesian_to_polar(pts):
        """Convert Cartesian points to polar coordinates."""
        x = pts[:, 0]
        y = pts[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        # Unwrap to [0, 2*pi)
        theta = theta % (2 * np.pi)
        # Interleave r and theta
        polar = np.zeros(2 * n)
        polar[0::2] = r
        polar[1::2] = theta
        return polar

    best_score = -np.inf
    best_points = None

    # Start with regular octagon
    oct_angles = 2 * np.pi * np.arange(n) / n
    oct_points = np.column_stack([np.cos(oct_angles), np.sin(oct_angles)])
    best_score = compute_score(oct_points)
    best_points = oct_points.copy()

    # Phase 1: Differential evolution in polar coordinate space
    # Bounds: r in [0.1, 2.0], theta in [0, 2*pi)
    r_bounds = (0.1, 2.0)
    theta_bounds = (0.0, 2.0 * np.pi)
    
    # Create bounds array: [r_min, r_max, theta_min, theta_max, ...]
    bounds = []
    for i in range(n):
        bounds.append(r_bounds)
        bounds.append(theta_bounds)
    
    # Run differential evolution with multiple seeds for robustness
    n_de_attempts = 3
    for attempt in range(n_de_attempts):
        if time.time() > deadline - 60:
            break
        
        seed = 42 + attempt
        try:
            de_result = differential_evolution(
                objective_polar,
                bounds=bounds,
                seed=seed,
                maxiter=150,
                popsize=15,
                tol=1e-8,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=True,
                init='latinhypercube'
            )
            
            if de_result.fun < 1e9:
                de_polar = de_result.x
                de_pts = polar_to_cartesian(de_polar)
                
                # Check distinctness
                dists_mat = np.sqrt(np.sum((de_pts[:, np.newaxis, :] - de_pts[np.newaxis, :, :])**2, axis=2))
                np.fill_diagonal(dists_mat, np.inf)
                if np.min(dists_mat) > 1e-8:
                    s_candidate = compute_score(de_pts)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_points = de_pts.copy()
        except Exception:
            pass

    # Phase 2: Local refinement using L-BFGS-B in Cartesian space
    if best_points is not None and time.time() < deadline - 30:
        try:
            lbfgs_result = minimize(
                full_objective_cartesian,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if lbfgs_result.fun < 1e9:
                refined_pts = lbfgs_result.x.reshape(8, 2)
                dists_mat = np.sqrt(np.sum((refined_pts[:, np.newaxis, :] - refined_pts[np.newaxis, :, :])**2, axis=2))
                np.fill_diagonal(dists_mat, np.inf)
                if np.min(dists_mat) > 1e-8:
                    s_candidate = compute_score(refined_pts)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_points = refined_pts.copy()
        except Exception:
            pass

    # Phase 3: Try perturbations from regular octagon with Nelder-Mead
    if time.time() < deadline - 20:
        for pert_scale in [0.05, 0.1, 0.2, 0.3]:
            if time.time() > deadline - 10:
                break
            np.random.seed(123 + int(pert_scale * 100))
            perturb = np.random.randn(16) * pert_scale
            oct_polar = cartesian_to_polar(oct_points)
            init_polar = oct_polar + perturb
            
            try:
                nm_result = minimize(
                    objective_polar,
                    init_polar,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                if nm_result.fun < 1e9:
                    nm_polar = nm_result.x
                    nm_pts = polar_to_cartesian(nm_polar)
                    dists_mat = np.sqrt(np.sum((nm_pts[:, np.newaxis, :] - nm_pts[np.newaxis, :, :])**2, axis=2))
                    np.fill_diagonal(dists_mat, np.inf)
                    if np.min(dists_mat) > 1e-8:
                        s_candidate = compute_score(nm_pts)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_points = nm_pts.copy()
            except Exception:
                pass

    # Phase 4: Powell refinement on best point found
    if best_points is not None and time.time() < deadline - 5:
        try:
            powell_result = minimize(
                full_objective_cartesian,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-16, 'xtol': 1e-14}
            )
            
            if powell_result.fun < 1e9:
                powell_pts = powell_result.x.reshape(8, 2)
                dists_mat = np.sqrt(np.sum((powell_pts[:, np.newaxis, :] - powell_pts[np.newaxis, :, :])**2, axis=2))
                np.fill_diagonal(dists_mat, np.inf)
                if np.min(dists_mat) > 1e-8:
                    s_candidate = compute_score(powell_pts)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_points = powell_pts.copy()
        except Exception:
            pass

    # Fallback to regular octagon if something went wrong
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    # Final check: distinct points
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord >= 1e6:
        scale = 1e6 / max_coord * 0.999
        best_points = best_points * scale

    return [[float(p[0]), float(p[1])] for p in best_points]
