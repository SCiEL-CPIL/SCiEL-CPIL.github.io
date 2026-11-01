"""Improved program for packing 300 circles in a unit circle using radial-ring decomposition with smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _ring_decomposition_init(n, num_rings=None, ring_spread=0.92):
    """
    Initialize points in concentric rings with optimized radial distances.
    Each ring gets a proportional number of points based on circumference.
    """
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2.0) + 3
    
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Estimate circle radius from area
    r_est = 1.0 / np.sqrt(n)
    
    # Ring radii
    ring_radii = [ring_spread * (i + 1) / (num_rings + 1) for i in range(num_rings)]
    
    # Distribute points proportionally to circumference
    circumferences = [2.0 * np.pi * rr for rr in ring_radii]
    total_circ = sum(circumferences)
    
    ring_counts = []
    for i, circ in enumerate(circumferences):
        count = int(remaining * circ / total_circ)
        ring_counts.append(max(1, count))
    
    # Adjust to get exact count
    diff = remaining - sum(ring_counts)
    if diff > 0:
        for _ in range(diff):
            ring_counts[-1] += 1
    elif diff < 0:
        for i in range(len(ring_counts) - 1, -1, -1):
            if ring_counts[i] > 1 and -diff > 0:
                ring_counts[i] -= 1
                diff += 1
    
    # Place points in each ring with angular offsets
    current_count = 1
    for i, (count, rr) in enumerate(zip(ring_counts, ring_radii)):
        if count == 0:
            continue
        d_theta = 2.0 * np.pi / count
        offset = np.pi / count * (i % 2)
        for k in range(count):
            theta = k * d_theta + offset
            points.append([rr * np.cos(theta), rr * np.sin(theta)])
        current_count += count
    
    points = np.array(points[:n])
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    Tries multiple spacings to find the best initial packing.
    """
    best_pts = None
    best_score = -1.0
    
    for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        spacing = 2.0 / (np.sqrt(n) * scale)
        
        nx = int(2.0 / spacing) + 3
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
        
        x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
        
        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                    pts_list.append([px, py])
        
        if len(pts_list) < n:
            continue
            
        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(boundary_dists)
        score = min(min_pair_dist, min_boundary)
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
    
    best_pts = best_pts + np.random.normal(0, 1e-4, best_pts.shape)
    
    return best_pts

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min and its gradient weights using log-sum-exp trick.
    soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
    Returns (soft_min_value, weights) where weights sum to 1.
    """
    values = np.asarray(values, dtype=np.float64)
    z = -beta * values
    
    max_z = np.max(z)
    # Clip z - max_z to avoid underflow issues for very negative values
    shifted = z - max_z
    shifted = np.clip(shifted, -700.0, 0.0)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pair contributions
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    grad_obj = -grad
    
    return objective, grad_obj

def _local_optimize(points, time_limit=30.0, beta_schedule=None, max_iter=500):
    """
    Local optimization using L-BFGS-B with soft-min objective and adaptive beta.
    """
    start_time = time.time()
    
    if beta_schedule is None:
        beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 50000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            if beta >= 50000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-11,
                    'ftol': 1e-15,
                    'maxls': 100,
                    'disp': False
                }
            elif beta >= 10000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxls': 100,
                    'disp': False
                }
            elif beta >= 1000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-9,
                    'ftol': 1e-13,
                    'maxls': 80,
                    'disp': False
                }
            else:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options=opts
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                
        except Exception:
            continue
    
    return current_points, current_score

def _force_directed_refine(points, iterations=200, time_limit=10.0):
    """
    Force-directed local optimization: push apart overlapping circles,
    pull toward boundary to maximize radius.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current minimum radius
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - norms)
        r = min(min_pair, min_boundary)
        
        # Build forces
        forces = np.zeros_like(points)
        
        # Pair repulsion for close pairs
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        i_idx, j_idx = iu
        
        # Only consider pairs within 2*r * 1.5
        close_mask = pair_dists < 2.0 * r * 1.5
        if np.any(close_mask):
            close_i = i_idx[close_mask]
            close_j = j_idx[close_mask]
            close_d = pair_dists[close_mask]
            vecs = points[close_i] - points[close_j]
            safe_d = np.where(close_d > 1e-10, close_d, 1e-10)
            # Force magnitude: stronger for closer pairs
            mag = (2.0 * r - close_d) / safe_d
            forces[close_i] += mag[:, np.newaxis] * vecs
            forces[close_j] -= mag[:, np.newaxis] * vecs
        
        # Boundary attraction: pull points toward boundary if they're the limiting factor
        if min_boundary <= min_pair:
            # Points near boundary get pushed inward slightly, others get pulled outward
            for i in range(n):
                norm_i = norms[i]
                if norm_i > 0.01:
                    # Radial direction
                    radial_dir = points[i] / norm_i
                    # If this point is close to being the boundary constraint, push inward
                    if 1.0 - norm_i < r * 1.1:
                        forces[i] -= radial_dir * 0.01
                    else:
                        # Pull slightly outward to use boundary
                        forces[i] += radial_dir * 0.005
        
        # Apply forces with adaptive step size
        step = 0.1 * r / (1.0 + it * 0.01)
        points += step * forces
        
        # Clamp to unit circle
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - 1e-10:
                points[i] = points[i] / norms[i] * (1.0 - 1e-10)
        
        # Check for improvement
        new_score = _compute_score(points)
        if new_score < r - 1e-8:
            # Revert
            points = points - step * forces
            norms = np.linalg.norm(points, axis=1)
            for i in range(n):
                if norms[i] > 1.0 - 1e-10:
                    points[i] = points[i] / norms[i] * (1.0 - 1e-10)
    
    return points, _compute_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try multiple initializations
    initializations = [
        ("hex", _hex_lattice_init(N)),
        ("ring_8", _ring_decomposition_init(N, 8, 0.92)),
        ("ring_10", _ring_decomposition_init(N, 10, 0.90)),
    ]
    
    best_points = None
    best_score = -np.inf
    
    for name, points in initializations:
        if time.time() > deadline - 60:
            break
        
        score = _compute_score(points)
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Optimization Stage 1: Coarse (Global-ish)
    if time.time() < deadline - 50:
        remaining_time = min(35.0, deadline - 50 - time.time())
        best_points, best_score = _local_optimize(
            best_points, 
            time_limit=remaining_time,
            beta_schedule=[10.0, 30.0, 100.0, 300.0],
            max_iter=150
        )
    
    # Optimization Stage 2: Fine (Local refinement)
    if time.time() < deadline - 25:
        remaining_time = min(25.0, deadline - 25 - time.time())
        best_points, best_score = _local_optimize(
            best_points,
            time_limit=remaining_time,
            beta_schedule=[1000.0, 3000.0, 10000.0],
            max_iter=150
        )
    
    # Optimization Stage 3: Force-directed refinement
    if time.time() < deadline - 15:
        remaining_time = min(12.0, deadline - 15 - time.time())
        if remaining_time > 2.0:
            fd_points, fd_score = _force_directed_refine(
                best_points,
                iterations=300,
                time_limit=remaining_time
            )
            if fd_score > best_score:
                best_points = fd_points
                best_score = fd_score
    
    # Optimization Stage 4: Polish (High precision)
    if time.time() < deadline - 8:
        remaining_time = min(15.0, deadline - 8 - time.time())
        best_points, best_score = _local_optimize(
            best_points,
            time_limit=remaining_time,
            beta_schedule=[50000.0, 100000.0, 200000.0],
            max_iter=200
        )
    
    # Optimization Stage 5: Final ultra-precision polish
    if time.time() < deadline - 3:
        remaining_time = min(5.0, deadline - 3 - time.time())
        if remaining_time > 1.0:
            best_points, best_score = _local_optimize(
                best_points,
                time_limit=remaining_time,
                beta_schedule=[500000.0, 1000000.0],
                max_iter=100
            )
    
    # Final validation and clamping
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-12)
        best_score = _compute_score(best_points)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.9, neginf=-0.9)
        best_score = _compute_score(best_points)
    
    # One final tiny polish pass if time permits
    if time.time() < deadline - 1.0:
        remaining_time = min(1.5, deadline - 1.0 - time.time())
        if remaining_time > 0.5:
            try:
                final_points, final_score = _local_optimize(
                    best_points,
                    time_limit=remaining_time,
                    beta_schedule=[2000000.0],
                    max_iter=50
                )
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
            except Exception:
                pass
    
    return best_points

# EVOLVE-BLOCK-END
