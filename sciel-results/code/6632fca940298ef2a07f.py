# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
import random
import sys

def _radial_warp(points, alpha, beta, gamma):
    """Apply a radial distortion: r' = r * (1 + alpha*r + beta*r^2) to each point."""
    r = np.sqrt(np.sum(points**2, axis=1, keepdims=True))
    factor = 1.0 + alpha * r + beta * r**2
    return points * factor

def _projective_warp_2d(points, a, b, c, d, e, f):
    """Apply a 2D projective (homography) transformation."""
    # H = [[a, b, c], [d, e, f], [1, 1, 1]] applied to (x, y, 1)
    denom = 1.0 + c * points[:, 0] + f * points[:, 1]
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    x_new = (a * points[:, 0] + b * points[:, 1] + c) / denom
    y_new = (d * points[:, 0] + e * points[:, 1] + f) / denom
    return np.column_stack([x_new, y_new])

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin

    n_points = 16

    def compute_score(points):
        if points.shape != (n_points, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        """Scale points such that the maximum pairwise distance is 1."""
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_objective(x, beta):
        """
        Objective function to minimize: -soft_min(distances).
        The points are normalized to unit diameter internally.
        """
        points = x.reshape(n_points, 2)
        points = normalize_points(points)
        d = pdist(points)
        if np.max(d) < 1e-12:
            return 0.0
        vals = np.exp(-beta * d)
        sum_vals = np.sum(vals)
        if sum_vals == 0:
            return 0.0
        soft_min_val = -1.0 / beta * np.log(sum_vals)
        return -soft_min_val

    def objective_value_only(x, beta):
        """Just the objective value for faster evaluation."""
        points = x.reshape(n_points, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        d_norm = d / d_max
        vals = np.exp(-beta * d_norm)
        sum_vals = np.sum(vals)
        if sum_vals == 0:
            return 0.0
        return -1.0 / beta * np.log(sum_vals)

    def optimize_candidate(points, time_budget=5.0, betas=None):
        """
        Optimizes a candidate using L-BFGS-B with increasing beta (temperature).
        """
        if betas is None:
            betas = [5, 10, 20, 50, 100, 200, 500]
            
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return objective_value_only(x, beta)
            
            try:
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 800}
                )
                
                # Check result
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                # Update current points for next beta stage
                points = best_points.copy()
                
            except Exception:
                pass
                
            # If time is running out, break
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    best_points = None
    best_score = -np.inf
    all_candidates = []

    # 1. Generate Diverse Initial Candidates
    
    # 1a. Triangular Lattice (Cluster)
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    all_candidates.append(lattice_points[idx])

    # 1b. Square Grid (4x4) - base for projective warp
    grid_coords = []
    for i in range(4):
        for j in range(4):
            grid_coords.append([i - 1.5, j - 1.5])
    base_grid = np.array(grid_coords)
    all_candidates.append(base_grid.copy())

    # 1b2. Radially warped 4x4 grid with various parameters
    # Try different radial warp strengths to equalize distances
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for beta_w in [0.0, -0.05, -0.1, 0.05, 0.1]:
            warped = _radial_warp(base_grid, alpha, beta_w, 0.0)
            # Center it
            warped = warped - np.mean(warped, axis=0)
            all_candidates.append(warped)

    # 1b3. Projective warps of 4x4 grid
    # Small projective distortions to break symmetry
    proj_params_list = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # identity
        (0.05, 0.0, 0.0, 0.0, 0.05, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.05),
        (0.05, 0.0, 0.0, 0.0, 0.05, 0.05),
        (0.1, 0.0, 0.0, 0.0, 0.1, 0.0),
        (0.0, 0.1, 0.0, 0.1, 0.0, 0.0),
        (0.05, 0.05, 0.0, 0.05, 0.05, 0.0),
        (-0.05, 0.0, 0.0, 0.0, -0.05, 0.0),
        (0.0, 0.0, 0.02, 0.0, 0.0, 0.02),
        (0.0, 0.0, -0.02, 0.0, 0.0, -0.02),
        (0.08, 0.0, 0.01, 0.0, 0.08, 0.01),
        (0.0, 0.08, 0.0, 0.08, 0.0, 0.01),
    ]
    for a, b, c, d, e, f in proj_params_list:
        warped = _projective_warp_2d(base_grid, a, b, c, d, e, f)
        warped = warped - np.mean(warped, axis=0)
        all_candidates.append(warped)

    # 1c. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    all_candidates.append(poly_points)

    # 1d. Concentric Circles (4 inner, 12 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    inner_radius = 0.4
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    all_candidates.append(np.vstack([inner_points, outer_points]))

    # 1e. Concentric Circles (6 inner, 10 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    inner_radius = 0.45
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    all_candidates.append(np.vstack([inner_points, outer_points]))

    # 1f. Random Points (Multiple seeds) - reduced to save time for new candidates
    for seed in range(20):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        random_points = random_points / np.max(pdist(random_points))
        all_candidates.append(random_points)

    # 1g. Jittered Lattice
    base_lattice = lattice_points[idx].copy()
    for j_seed in range(5):
        np.random.seed(1000 + j_seed)
        jittered = base_lattice + np.random.randn(n_points, 2) * 0.05
        all_candidates.append(jittered)

    # 1h. Hexagonal arrangement
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_points = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    center_point = np.array([[0.0, 0.0]])
    ring2_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring2_points = np.column_stack([2.0 * np.cos(ring2_angles), 2.0 * np.sin(ring2_angles)])
    all_candidates.append(np.vstack([hex_points, center_point, ring2_points[:9]]))

    # 1i. Fibonacci spiral distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    spiral_points = []
    for i in range(16):
        r = np.sqrt(i + 1) / 4
        theta = i * golden_angle
        spiral_points.append([r * np.cos(theta), r * np.sin(theta)])
    all_candidates.append(np.array(spiral_points))

    # 1j. Dedicated radial warp optimization on 4x4 grid
    # Optimize (alpha, beta) parameters for the radial warp
    def radial_warp_objective(params):
        alpha, beta_w = params
        warped = _radial_warp(base_grid, alpha, beta_w, 0.0)
        warped = warped - np.mean(warped, axis=0)
        d = pdist(warped)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(d)
        return -(d_min / d_max) ** 2

    # Multi-start radial warp optimization
    radial_starts = [
        [0.0, 0.0], [0.2, 0.0], [0.3, -0.1], [0.4, -0.15],
        [0.5, -0.2], [0.1, 0.05], [0.3, 0.1], [0.6, -0.25]
    ]
    for rs in radial_starts:
        if time.time() > deadline - 60:
            break
        try:
            res_rw = minimize(
                radial_warp_objective,
                np.array(rs, dtype=float),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_rw.fun < -0.5:  # Only accept if score > 0.5
                alpha_opt, beta_opt = res_rw.x
                warped = _radial_warp(base_grid, alpha_opt, beta_opt, 0.0)
                warped = warped - np.mean(warped, axis=0)
                all_candidates.append(warped)
        except Exception:
            pass

    # 1k. Dedicated projective warp optimization on 4x4 grid
    def projective_warp_objective(params):
        a, b, c, d, e, f = params
        warped = _projective_warp_2d(base_grid, a, b, c, d, e, f)
        warped = warped - np.mean(warped, axis=0)
        d = pdist(warped)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(d)
        return -(d_min / d_max) ** 2

    proj_starts = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.05, 0.05, 0.0, 0.05, 0.05, 0.0],
        [0.0, 0.0, 0.02, 0.0, 0.0, 0.02],
        [0.08, 0.0, 0.01, 0.0, 0.08, 0.01],
        [0.0, 0.08, 0.0, 0.08, 0.0, 0.01],
    ]
    for ps in proj_starts:
        if time.time() > deadline - 60:
            break
        try:
            res_pw = minimize(
                projective_warp_objective,
                np.array(ps, dtype=float),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_pw.fun < -0.5:
                warped = _projective_warp_2d(base_grid, *res_pw.x)
                warped = warped - np.mean(warped, axis=0)
                all_candidates.append(warped)
        except Exception:
            pass

    # 2. Evaluate and Optimize Candidates in Batches
    
    # Sort candidates by initial score to prioritize good ones
    scored_candidates = []
    for cand in all_candidates:
        s = compute_score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    processed = []
    
    # Budget allocation: top candidates get more time
    # We have ~60-80 candidates now. Allocate time carefully.
    # Top 3: 12s each = 36s
    # Next 7: 8s each = 56s
    # Next 10: 5s each = 50s
    # Rest: 3s each
    # Total for first 20: ~142s, leaving ~450s for rest + polishing
    
    for i, (init_score, cand) in enumerate(scored_candidates):
        if time.time() > deadline - 20:
            break
        
        # Determine budget based on rank
        if i < 3:
            budget = 12.0
        elif i < 10:
            budget = 8.0
        elif i < 20:
            budget = 5.0
        else:
            budget = 3.0
            
        opt_cand, opt_score = optimize_candidate(cand, time_budget=budget)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Refine the top candidates with higher precision
    
    top_candidates = processed[:5]
    for score, cand in top_candidates:
        if time.time() > deadline - 10:
            break
            
        def optimize_candidate_high(cand, time_budget=20.0):
            points = cand.copy()
            points = normalize_points(points)
            best_points = points.copy()
            best_score = compute_score(best_points)
            
            betas = [100, 200, 500, 1000, 2000, 5000]
            start_opt = time.time()
            
            for beta in betas:
                if time.time() - start_opt > time_budget:
                    break
                
                def obj(x):
                    return objective_value_only(x, beta)
                
                try:
                    res = minimize(
                        obj, 
                        points.flatten(), 
                        method='L-BFGS-B', 
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
                    )
                    
                    opt_points = res.x.reshape(n_points, 2)
                    opt_points = normalize_points(opt_points)
                    opt_score = compute_score(opt_points)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                    points = best_points.copy()
                    
                except Exception:
                    pass
                
                if time.time() - start_opt > time_budget:
                    break
            return best_points, best_score

        opt_cand, opt_score = optimize_candidate_high(cand, time_budget=20.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # 4. Final Polish: Nelder-Mead on the very best point
    if best_points is not None and time.time() < deadline - 15:
        def nm_objective(x):
            points = x.reshape(n_points, 2)
            points = normalize_points(points)
            d = pdist(points)
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                return 0.0
            return -(d_min / d_max) ** 2
        
        bp = normalize_points(best_points).flatten()
        
        try:
            res = minimize(
                nm_objective, 
                bp, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            nm_points = res.x.reshape(n_points, 2)
            nm_points = normalize_points(nm_points)
            nm_score = compute_score(nm_points)
            
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
                
        except Exception:
            pass

    # 5. Second round of radial/projective warp refinement on best configuration
    if best_points is not None and time.time() < deadline - 20:
        # Use the best found configuration as a new base for warp optimization
        bp_centered = best_points - np.mean(best_points, axis=0)
        
        def radial_warp_from_best(params):
            alpha, beta_w = params
            warped = _radial_warp(bp_centered, alpha, beta_w, 0.0)
            d = pdist(warped)
            d_max = np.max(d)
            if d_max < 1e-12:
                return 0.0
            d_min = np.min(d)
            return -(d_min / d_max) ** 2
        
        # Try small perturbations around identity
        for start_params in [[0.0, 0.0], [0.05, 0.0], [0.1, -0.05], [-0.05, 0.05]]:
            if time.time() > deadline - 10:
                break
            try:
                res_rw2 = minimize(
                    radial_warp_from_best,
                    np.array(start_params, dtype=float),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if res_rw2.fun < -best_score:  # Better than current best
                    alpha_opt, beta_opt = res_rw2.x
                    warped = _radial_warp(bp_centered, alpha_opt, beta_opt, 0.0)
                    warped = normalize_points(warped)
                    s = compute_score(warped)
                    if s > best_score:
                        best_score = s
                        best_points = warped.copy()
            except Exception:
                pass

    # Ensure valid output
    if best_points is None:
        best_points = all_candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
