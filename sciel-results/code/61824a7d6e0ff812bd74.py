"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DEADLINE_SECONDS = 115.0  # Safety margin before 150s limit

def compute_radius(pts):
    """
    Computes the maximum radius for the given centers.
    pts: (N, 2) array of centers.
    Returns r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return r

def lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset=0.0):
    """
    Generate points from a parametric hexagonal lattice.
    row_lengths: list of ints, number of circles per row
    h_spacing: horizontal spacing between centers in a row
    v_spacing: vertical spacing between rows
    x_offsets: list of x-offsets for each row (relative to center)
    y_offset: global vertical shift
    Returns (N, 2) array.
    """
    points = []
    n_rows = len(row_lengths)
    total_height = (n_rows - 1) * v_spacing
    y_start = -total_height / 2.0 + y_offset
    
    for i, length in enumerate(row_lengths):
        y = y_start + i * v_spacing
        total_width = (length - 1) * h_spacing
        x_start = -total_width / 2.0 + x_offsets[i]
        
        for j in range(length):
            x = x_start + j * h_spacing
            points.append([x, y])
    
    pts = np.array(points)
    return pts

def soft_min(vals, beta=50.0):
    """
    Smooth approximation of min using log-sum-exp.
    As beta -> inf, soft_min -> exact min.
    """
    vals = np.asarray(vals)
    m = np.min(vals)
    if m < -1e6:
        return m
    # Use stable log-sum-exp: m - log(sum(exp(beta*(v-m)))) / beta
    # For min: soft_min = -log(sum(exp(-beta*v)))/beta
    # More stable: shift by min
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def smooth_objective(params, row_lengths, beta=50.0):
    """
    Smooth objective using soft-min surrogate.
    params: [h_spacing, v_spacing, x_off_0, ..., y_offset]
    Returns negative smooth radius (to minimize).
    """
    n_rows = len(row_lengths)
    h_spacing = params[0]
    v_spacing = params[1]
    x_offsets = params[2:2+n_rows]
    y_offset = params[2+n_rows]
    
    if h_spacing <= 0.01 or v_spacing <= 0.01:
        return 100.0
    
    pts = lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset)
    
    # Boundary distances
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pairwise_dists = np.sqrt(dist_sq) / 2.0
    # Remove diagonal
    pairwise_dists = pairwise_dists[~np.eye(N, dtype=bool)].reshape(-1)
    
    # Combine all constraints into one vector
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Use soft-min for smoothness
    r_smooth = soft_min(all_dists, beta=beta)
    
    # Penalize if any point is far outside
    min_bound_exact = np.min(boundary_dists)
    if min_bound_exact < -0.05:
        return 100.0 + abs(min_bound_exact) * 10
    
    return -r_smooth

def optimize_lattice(row_lengths, start_time, deadline):
    """
    Optimize lattice parameters for a given row-length pattern.
    Uses smooth objective with increasing beta, then exact refinement.
    """
    n_rows = len(row_lengths)
    n_params = 2 + n_rows + 1  # h_spacing, v_spacing, x_offsets..., y_offset
    
    # Wider bounds for more flexibility
    bounds = [(0.06, 0.22), (0.06, 0.22)] + [(-0.08, 0.08)] * n_rows + [(-0.08, 0.08)]
    
    # Initial guess: hexagonal packing
    h0 = 1.0 / (max(row_lengths) + 0.5)
    v0 = h0 * np.sqrt(3) / 2.0
    x_offs0 = [0.0] * n_rows
    for i in range(1, n_rows, 2):
        x_offs0[i] = h0 / 2.0
    y0 = 0.0
    
    x0 = np.array([h0, v0] + x_offs0 + [y0])
    
    x_best = x0.copy()
    
    # Phase 1: Coarse search with smooth objective (low beta)
    if time.time() < deadline - 25:
        try:
            de_result = differential_evolution(
                smooth_objective,
                bounds,
                args=(row_lengths, 20.0),
                maxiter=100,
                popsize=12,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            x_best = de_result.x
        except Exception:
            pass
    
    # Phase 2: Medium precision with higher beta
    if time.time() < deadline - 15:
        try:
            nm_result = minimize(
                smooth_objective,
                x_best,
                args=(row_lengths, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            x_best = nm_result.x
        except Exception:
            pass
    
    # Phase 3: High precision with very high beta (near-exact)
    if time.time() < deadline - 8:
        try:
            nm_result = minimize(
                smooth_objective,
                x_best,
                args=(row_lengths, 200.0),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x_best = nm_result.x
        except Exception:
            pass
    
    # Phase 4: Exact objective refinement
    if time.time() < deadline - 3:
        try:
            nm_result = minimize(
                lattice_objective,
                x_best,
                args=(row_lengths,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            x_best = nm_result.x
        except Exception:
            pass
    
    # Extract final parameters
    h_spacing = x_best[0]
    v_spacing = x_best[1]
    x_offsets = x_best[2:2+n_rows]
    y_offset = x_best[2+n_rows]
    
    pts = lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset)
    pts = np.clip(pts, -0.4999, 0.4999)
    r = compute_radius(pts)
    
    return pts, r

def smooth_free_objective(x, beta=50.0):
    """
    Smooth objective for free-point optimization using soft-min.
    """
    p = x.reshape(-1, 2)
    d_x = 0.5 - np.abs(p[:, 0])
    d_y = 0.5 - np.abs(p[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pairwise_dists = np.sqrt(dist_sq) / 2.0
    pairwise_dists = pairwise_dists[~np.eye(N, dtype=bool)].reshape(-1)
    
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    r_smooth = soft_min(all_dists, beta=beta)
    
    min_bound_exact = np.min(boundary_dists)
    if min_bound_exact < -0.05:
        return 100.0 + abs(min_bound_exact) * 10
    
    return -r_smooth

def free_point_optimize(pts, start_time, deadline):
    """
    Refine free points using multi-stage smooth optimization.
    """
    def neg_r_exact(x):
        p = x.reshape(-1, 2)
        d_x = 0.5 - np.abs(p[:, 0])
        d_y = 0.5 - np.abs(p[:, 1])
        min_bound = np.min(np.minimum(d_x, d_y))
        if min_bound < 0:
            return -min_bound * 10 + 100.0
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_inter = np.sqrt(np.min(dist_sq))
        r = np.minimum(min_dist_inter / 2.0, min_bound)
        return -r
    
    current_pts = pts.copy()
    
    # Stage 1: Coarse smooth optimization
    if time.time() < deadline - 20:
        try:
            result = minimize(
                smooth_free_objective,
                current_pts.flatten(),
                args=(20.0,),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-6, 'ftol': 1e-8}
            )
            current_pts = result.x.reshape(-1, 2)
            current_pts = np.clip(current_pts, -0.4999, 0.4999)
        except Exception:
            pass
    
    # Stage 2: Medium precision
    if time.time() < deadline - 12:
        try:
            result = minimize(
                smooth_free_objective,
                current_pts.flatten(),
                args=(50.0,),
                method='Powell',
                options={'maxiter': 80, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            current_pts = result.x.reshape(-1, 2)
            current_pts = np.clip(current_pts, -0.4999, 0.4999)
        except Exception:
            pass
    
    # Stage 3: High precision smooth
    if time.time() < deadline - 6:
        try:
            result = minimize(
                smooth_free_objective,
                current_pts.flatten(),
                args=(200.0,),
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            current_pts = result.x.reshape(-1, 2)
            current_pts = np.clip(current_pts, -0.4999, 0.4999)
        except Exception:
            pass
    
    # Stage 4: Exact objective polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                neg_r_exact,
                current_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            current_pts = result.x.reshape(-1, 2)
            current_pts = np.clip(current_pts, -0.4999, 0.4999)
        except Exception:
            pass
    
    return current_pts, compute_radius(current_pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pts = None
    best_r = -1.0
    
    # Multiple row-length patterns that sum to 50, ordered by expected quality
    patterns = [
        [8, 7, 7, 7, 7, 7, 7],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [7, 6, 7, 7, 7, 7, 6, 3],
        [6, 7, 7, 7, 7, 7, 6, 3],
        [8, 8, 7, 7, 7, 7, 6],
        [6, 6, 7, 7, 7, 7, 7, 3],
        [7, 7, 7, 7, 7, 7, 6, 2],
        [5, 7, 7, 7, 7, 7, 7, 3],
        [7, 7, 7, 7, 7, 6, 7, 2],
        [6, 7, 6, 7, 7, 7, 7, 3],
        [7, 7, 6, 7, 7, 7, 6, 3],
    ]
    
    # Filter patterns that sum to N
    valid_patterns = [p for p in patterns if sum(p) == N]
    
    # If no valid patterns, create some
    if not valid_patterns:
        valid_patterns = [[8, 7, 7, 7, 7, 7, 7], [7, 6, 7, 6, 7, 6, 7, 4]]
    
    # Limit number of patterns to fit within time budget
    max_patterns = 6
    
    for idx, pattern in enumerate(valid_patterns[:max_patterns]):
        if time.time() > deadline - 25:
            break
        
        pts, r = optimize_lattice(pattern, start_time, deadline)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Additional refinement with free-point optimization
        if time.time() < deadline - 15:
            pts_ref, r_ref = free_point_optimize(pts, start_time, deadline)
            if r_ref > best_r:
                best_r = r_ref
                best_pts = pts_ref.copy()
    
    # Also try a few random initializations with free-point optimization
    if time.time() < deadline - 18:
        for trial in range(3):
            if time.time() > deadline - 12:
                break
            rand_pts = np.random.uniform(-0.38, 0.38, size=(N, 2))
            pts_ref, r_ref = free_point_optimize(rand_pts, start_time, deadline)
            if r_ref > best_r:
                best_r = r_ref
                best_pts = pts_ref.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 3:
        try:
            pts_final, r_final = free_point_optimize(best_pts, start_time, deadline)
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final.copy()
        except Exception:
            pass
    
    # Final safety
    if best_pts is None or best_r < 0:
        # Fallback: simple grid
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_r = compute_radius(best_pts)
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts

# EVOLVE-BLOCK-END
