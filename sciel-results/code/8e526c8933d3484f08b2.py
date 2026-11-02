"""
Packing of 50 equal circles in a unit square.
Strategy: Iterative radius increase with L-BFGS polishing and Simulated Annealing for escape.
Uses soft-min surrogate for smoother optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
# Precompute upper triangle indices for pairwise distances
IU = np.triu_indices(N, k=1)
SOFT_BETA = 20.0  # Temperature for soft-min

def _compute_r(pts):
    """
    Compute the maximum common radius for the given configuration.
    pts: (N, 2) array of centers.
    Returns: float radius.
    """
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    
    # Distance to walls
    d_walls = np.minimum(
        np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1])
    )
    r_wall = np.min(d_walls)
    
    # Distance between circles using cKDTree for efficiency
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    r_pair = np.min(dists[:, 1]) / 2.0
    
    return float(min(r_wall, r_pair))

def _soft_min_obj(x_flat, r_target, beta):
    """
    Soft-min objective: -soft_min of all constraints (pair distances and wall distances).
    We minimize this to maximize the minimum constraint.
    soft_min(a_i) ≈ -(1/beta) * log(sum(exp(-beta * a_i)))
    """
    pts = x_flat.reshape(N, 2)
    
    # Wall distances
    d_left = HALF + pts[:, 0]
    d_right = HALF - pts[:, 0]
    d_bottom = HALF + pts[:, 1]
    d_top = HALF - pts[:, 1]
    
    # Pair distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e15)
    pair_dists = dists[IU]
    
    # All constraint values (we want to maximize the minimum)
    # Constraints: pair_dists/2 (radius limit from pairs), d_walls (radius limit from walls)
    all_constraints = np.concatenate([pair_dists / 2.0, d_left, d_right, d_bottom, d_top])
    
    # Soft-min: we want to maximize min(all_constraints)
    # Equivalent to minimizing -soft_min(all_constraints)
    # soft_min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    # To avoid overflow, use log-sum-exp trick
    max_c = np.max(all_constraints)
    log_sum = np.log(np.sum(np.exp(-beta * (all_constraints - max_c)))) + beta * max_c
    soft_min_val = -(1.0 / beta) * log_sum + max_c  # This is an approximation
    
    # Actually, let's use a cleaner formulation:
    # We want to maximize t such that all constraints >= t
    # Soft version: maximize - (1/beta) * log(sum(exp(-beta * (c_i - t))))
    # But t is not in x. Let's just minimize the negative soft-min of constraints.
    
    # Simpler: minimize sum of violations with soft penalty
    # Or: minimize -soft_min(all_constraints)
    
    # Let's use: obj = -soft_min(all_constraints)
    # soft_min(all_constraints) = -(1/beta) * log(sum(exp(-beta * all_constraints)))
    # To compute stably:
    max_val = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_val))
    log_sum = np.log(np.sum(exp_terms)) - beta * max_val
    soft_min_val = -(1.0 / beta) * log_sum
    
    return -soft_min_val

def _violation_obj_and_grad(x_flat, r_target):
    """
    Calculate the sum of squared violations and its gradient.
    Violation for pair (i,j): max(0, 2*r_target - d_ij)^2
    Violation for wall i: max(0, r_target - d_i)^2
    """
    pts = x_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e15)
    
    # Extract upper triangle distances and differences
    pair_dists = dists[IU]
    # We need the difference vectors for the gradient
    # diff[i, j] = pts[i] - pts[j]
    pair_diffs = diff[IU[0], IU[1], :]
    
    # Pair violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - pair_dists)
    obj = np.sum(pair_viol ** 2)
    
    # Gradient for pairs
    # d/dx_i sum (2r - d_ij)^2 = -2(2r - d_ij) * (x_i - x_j) / d_ij
    # Let v = 2r - d. If v > 0, grad_i += -2 * v * (x_i - x_j) / d
    # Note: The term appears for i and j.
    # For pair (i, j):
    # grad_i += -2 * v * (x_i - x_j) / d
    # grad_j += +2 * v * (x_i - x_j) / d  (since x_j - x_i = -(x_i - x_j))
    
    grad = np.zeros_like(pts)
    
    active = pair_viol > 0
    if np.any(active):
        v_active = pair_viol[active]
        d_active = pair_dists[active]
        diff_active = pair_diffs[active]
        
        # Factor: -2 * v / d
        factor = -2.0 * v_active / d_active
        
        # Scatter addition
        # For each active pair (i, j):
        # grad[i] += factor * (pts[i] - pts[j])
        # grad[j] += -factor * (pts[i] - pts[j])
        
        i_idx = IU[0][active]
        j_idx = IU[1][active]
        
        # Vectorized scatter add is tricky, use loops if N is small (50 is small)
        # Or use np.add.at
        force = factor[:, None] * diff_active
        
        np.add.at(grad, i_idx, force)
        np.add.at(grad, j_idx, -force)

    # Wall violations
    # d_left = HALF + pts[:, 0]
    # d_right = HALF - pts[:, 0]
    # d_bottom = HALF + pts[:, 1]
    # d_top = HALF - pts[:, 1]
    
    d_left = HALF + pts[:, 0]
    d_right = HALF - pts[:, 0]
    d_bottom = HALF + pts[:, 1]
    d_top = HALF - pts[:, 1]
    
    v_left = np.maximum(0.0, r_target - d_left)
    v_right = np.maximum(0.0, r_target - d_right)
    v_bottom = np.maximum(0.0, r_target - d_bottom)
    v_top = np.maximum(0.0, r_target - d_top)
    
    obj += np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
    
    # Gradients
    # d/dx (r - (HALF+x))^2 = -2(r - (HALF+x))
    # d/dx (r - (HALF-x))^2 = +2(r - (HALF-x))
    
    grad[:, 0] += -2.0 * v_left
    grad[:, 0] += 2.0 * v_right
    grad[:, 1] += -2.0 * v_bottom
    grad[:, 1] += 2.0 * v_top
    
    return obj, grad.flatten()

def _soft_min_obj_and_grad(x_flat, beta):
    """
    Soft-min objective and gradient.
    We minimize -soft_min(all_constraints) to maximize the minimum constraint.
    """
    pts = x_flat.reshape(N, 2)
    
    # Wall distances
    d_left = HALF + pts[:, 0]
    d_right = HALF - pts[:, 0]
    d_bottom = HALF + pts[:, 1]
    d_top = HALF - pts[:, 1]
    
    # Pair distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e15)
    pair_dists = dists[IU]
    
    # All constraint values
    # For pairs: constraint is d_ij / 2 (the radius allowed by pair separation)
    # For walls: constraint is d_wall (the radius allowed by wall distance)
    all_constraints = np.concatenate([pair_dists / 2.0, d_left, d_right, d_bottom, d_top])
    
    # soft_min(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # To compute stably:
    max_val = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_val))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = -(1.0 / beta) * log_sum
    
    obj = -soft_min_val
    
    # Gradient of -soft_min:
    # d/dx_i [-soft_min] = d/dx_i [(1/beta) * log(sum(exp(-beta * c_j)))]
    # = (1/beta) * [sum_j (-beta * exp(-beta * c_j)) / sum_k exp(-beta * c_k)] * d c_j / d x_i
    # = -sum_j [exp(-beta * c_j) / sum_k exp(-beta * c_k)] * d c_j / d x_i
    # Let w_j = exp(-beta * c_j) / sum_k exp(-beta * c_k)
    # Then grad = -sum_j w_j * (d c_j / d x_i)
    
    w = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(pts)
    
    # Pair constraints: c_ij = d_ij / 2
    # d c_ij / d x_i = (x_i - x_j) / (2 * d_ij)
    # d c_ij / d x_j = (x_j - x_i) / (2 * d_ij)
    
    pair_weights = w[:len(pair_dists)]
    
    active_pairs = np.where(pair_dists < 1e10)[0]  # Exclude diagonal
    if len(active_pairs) > 0:
        for idx in active_pairs:
            i = IU[0][idx]
            j = IU[1][idx]
            w_ij = pair_weights[idx]
            d_ij = pair_dists[idx]
            if d_ij < 1e-10:
                continue
            # d c_ij / d x_i = (x_i - x_j) / (2 * d_ij)
            # grad_i += -w_ij * (x_i - x_j) / (2 * d_ij)
            # grad_j += -w_ij * (x_j - x_i) / (2 * d_ij) = +w_ij * (x_i - x_j) / (2 * d_ij)
            factor = w_ij / (2.0 * d_ij)
            diff_ij = pts[i] - pts[j]
            grad[i] -= factor * diff_ij
            grad[j] += factor * diff_ij
    
    # Wall constraints
    # c_left = HALF + x_i, d c_left / d x_i = 1
    # c_right = HALF - x_i, d c_right / d x_i = -1
    # c_bottom = HALF + y_i, d c_bottom / d y_i = 1
    # c_top = HALF - y_i, d c_top / d y_i = -1
    
    wall_weights = w[len(pair_dists):]
    w_left = wall_weights[:N]
    w_right = wall_weights[N:2*N]
    w_bottom = wall_weights[2*N:3*N]
    w_top = wall_weights[3*N:4*N]
    
    # grad_x -= w_left (from -w_left * 1)
    # grad_x += w_right (from -w_right * (-1))
    # grad_y -= w_bottom
    # grad_y += w_top
    
    grad[:, 0] -= w_left
    grad[:, 0] += w_right
    grad[:, 1] -= w_bottom
    grad[:, 1] += w_top
    
    return obj, grad.flatten()

def _polish(pts, r_target, deadline, max_iter=500, use_soft_min=False):
    """
    Run L-BFGS-B to minimize violations for a given target radius.
    If use_soft_min, use the soft-min objective instead.
    """
    x0 = pts.flatten()
    
    # Bounds to keep points inside the square (with small margin)
    margin = 1e-9
    bounds = [(-HALF + margin, HALF - margin)] * (2 * N)
    
    try:
        if use_soft_min:
            res = minimize(
                lambda x: _soft_min_obj_and_grad(x, SOFT_BETA)[0],
                x0,
                jac=lambda x: _soft_min_obj_and_grad(x, SOFT_BETA)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
        else:
            res = minimize(
                lambda x: _violation_obj_and_grad(x, r_target)[0],
                x0,
                jac=lambda x: _violation_obj_and_grad(x, r_target)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _sa_step(pts, t, step_scale, rng):
    """
    Perform a single simulated annealing step.
    """
    new_pts = pts.copy()
    # Perturb 1-3 random points
    n_perturb = rng.integers(1, 4)
    idxs = rng.choice(N, size=n_perturb, replace=False)
    
    step_size = step_scale * np.sqrt(t)
    for idx in idxs:
        direction = rng.normal(0, 1, size=2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        new_pts[idx] += direction * step_size
    
    # Clip to bounds
    new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)
    return new_pts

def _hex_init():
    """Hexagonal close-packed initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    return np.array(pts[:N])

def _grid_init():
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take first 50
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with hex init
    best_pts = _hex_init()
    current_r = _compute_r(best_pts)
    
    # Initial target radius slightly higher than current
    r_target = current_r * 1.01 + 0.001
    
    # Main optimization loop
    # We alternate between polishing (L-BFGS) and annealing (SA)
    # If polishing fails to reduce violation, we anneal to escape.
    
    t_sa = 0.05
    t_min = 0.0001
    cooling = 0.999
    
    # First, do a quick polish to get a valid starting point
    best_pts = _polish(best_pts, r_target, deadline, max_iter=200)
    current_r = _compute_r(best_pts)
    
    stuck_count = 0
    soft_min_phase = False
    soft_min_iter = 0
    
    while time.time() < deadline - 10.0:
        # 1. Polish with L-BFGS
        # Use soft-min for the first few iterations to get a smooth landscape
        use_soft = soft_min_phase and soft_min_iter < 5
        polished_pts = _polish(best_pts, r_target, deadline, max_iter=300, use_soft_min=use_soft)
        if use_soft:
            soft_min_iter += 1
        r_achieved = _compute_r(polished_pts)
        
        # Check if we improved the radius
        if r_achieved > current_r + 1e-7:
            current_r = r_achieved
            best_pts = polished_pts
            stuck_count = 0
            # Increase target radius
            r_target = current_r * 1.005 + 0.0005
            # Cool down SA
            t_sa *= 0.95
            soft_min_phase = False  # Turn off soft-min when making progress
        else:
            # Check violation to see if we are stuck
            # If violation is low but radius didn't increase, maybe target is too high
            # Or we are in a local minimum.
            obj, _ = _violation_obj_and_grad(best_pts.flatten(), r_target)
            
            if obj < 1e-6:
                # Very small violation, likely feasible, try to increase target
                r_target += 0.0005
                stuck_count = 0
            else:
                # Stuck. Anneal.
                stuck_count += 1
                if stuck_count > 10:
                    # Reset temperature if stuck for too long
                    t_sa = 0.05
                    stuck_count = 0
                    # Enable soft-min phase to escape local minima
                    soft_min_phase = True
                    soft_min_iter = 0
                
                # Perform SA steps
                # Number of steps depends on time
                steps = 10
                for _ in range(steps):
                    if time.time() > deadline - 10.0:
                        break
                    new_pts = _sa_step(best_pts, t_sa, 0.02, rng)
                    r_new = _compute_r(new_pts)
                    
                    # Accept if better or with probability exp(-delta/t)
                    if r_new > current_r:
                        best_pts = new_pts
                        current_r = r_new
                    else:
                        delta = r_new - current_r
                        if delta < 0:
                            prob = np.exp(delta / t_sa)
                            if rng.random() < prob:
                                best_pts = new_pts
                                current_r = r_new
                
                t_sa *= cooling
                if t_sa < t_min:
                    t_sa = t_min

    # Final Polish with tighter tolerance
    if time.time() < deadline - 2.0:
        # Use soft-min for final polish to get a smoother convergence
        final_pts = _polish(best_pts, current_r, deadline, max_iter=3000, use_soft_min=True)
        r_final = _compute_r(final_pts)
        if r_final > current_r:
            best_pts = final_pts
            current_r = r_final
        
        # Then do a standard polish on top
        if time.time() < deadline - 1.0:
            final_pts2 = _polish(best_pts, current_r, deadline, max_iter=2000, use_soft_min=False)
            r_final2 = _compute_r(final_pts2)
            if r_final2 > current_r:
                best_pts = final_pts2
                current_r = r_final2

    return best_pts
