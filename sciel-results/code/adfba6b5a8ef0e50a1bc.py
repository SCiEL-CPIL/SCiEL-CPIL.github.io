"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, least_squares
import time
import random
import math


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _greedy_insertion(xy, r):
    """
    Greedy: given a set of circles, try to insert a new circle at the best spot
    that maximizes its radius without overlapping. Returns (new_xy, new_r, success).
    """
    n = len(xy)
    if n == 0:
        # First circle: center of square
        return np.array([[0.5, 0.5]]), np.array([0.5]), True
    
    # Candidate positions: grid + random
    grid_size = 40
    gx = np.linspace(0.01, 0.99, grid_size)
    gy = np.linspace(0.01, 0.99, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    
    # Also add some random candidates
    rng = np.random.default_rng(123)
    rand_pts = rng.uniform(0.01, 0.99, size=(200, 2))
    candidates = np.vstack([candidates, rand_pts])
    
    M = len(candidates)
    best_r = 0.0
    best_idx = -1
    
    for idx in range(M):
        cx, cy = candidates[idx, 0], candidates[idx, 1]
        # Max radius at this point: min distance to walls and to other circles
        wall_r = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if wall_r < 0.001:
            continue
        circle_r = wall_r
        for i in range(n):
            dx = cx - xy[i, 0]
            dy = cy - xy[i, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            max_r_at_point = dist - r[i]
            if max_r_at_point < circle_r:
                circle_r = max_r_at_point
            if circle_r < 0.001:
                break
        if circle_r > best_r:
            best_r = circle_r
            best_idx = idx
    
    if best_r < 0.005:
        return xy, r, False
    
    new_xy = np.vstack([xy, candidates[best_idx:best_idx+1]])
    new_r = np.concatenate([r, [best_r]])
    return new_xy, new_r, True


def _build_contact_graph(xy, r, tol=0.02):
    """
    Build a contact graph: edges between circles that are nearly touching,
    and between circles and walls.
    Returns list of (i, j, type) where type is 'circle' or 'wall_x'/'wall_y'.
    """
    n = len(xy)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if abs(dist - (r[i] + r[j])) < tol:
                contacts.append((i, j, 'circle'))
        # Wall contacts
        if abs(xy[i, 0] - r[i]) < tol:
            contacts.append((i, -1, 'wall_left'))
        if abs(1.0 - xy[i, 0] - r[i]) < tol:
            contacts.append((i, -1, 'wall_right'))
        if abs(xy[i, 1] - r[i]) < tol:
            contacts.append((i, -1, 'wall_bottom'))
        if abs(1.0 - xy[i, 1] - r[i]) < tol:
            contacts.append((i, -1, 'wall_top'))
    return contacts


def _contact_graph_optimize(xy, r, deadline):
    """
    Optimize the packing using a contact-graph-based barrier objective.
    We minimize a barrier function that penalizes violations of:
    1. Wall constraints (circle inside square)
    2. Non-overlap constraints (circles disjoint)
    3. Encourages contact (to make the packing rigid)
    
    Uses scipy.optimize.minimize with L-BFGS-B.
    """
    n = len(xy)
    start_time = time.time()
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    def _barrier_objective(params):
        params = np.asarray(params)
        xy = params[0::3].reshape(-1, 2)
        r = np.maximum(params[2::3], 0.001)
        
        # Wall barrier: penalty for going outside
        wall_penalty = 0.0
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            # Soft barrier: use log barrier
            for margin in [x - ri, 1.0 - x - ri, y - ri, 1.0 - y - ri]:
                if margin < 0.01:
                    wall_penalty += (0.01 - margin) ** 2
                else:
                    wall_penalty += -0.1 * math.log(margin)
        
        # Non-overlap barrier
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist + 0.01:
                    gap = dist - min_dist
                    if gap < 0.01:
                        overlap_penalty += (0.01 - gap) ** 2
                    else:
                        overlap_penalty += -0.05 * math.log(gap)
        
        # Encourage contact: reward when circles are close to touching
        contact_reward = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                gap = dist - min_dist
                if 0.0 < gap < 0.05:
                    contact_reward += 0.01 * (0.05 - gap)
        
        total_radius = np.sum(r)
        return -total_radius + wall_penalty + overlap_penalty - contact_reward
    
    try:
        result = minimize(
            _barrier_objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[2::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        return opt_xy, opt_r, total, valid
        
    except Exception:
        return xy, r, 0.0, False


def _radial_sweep_optimize(xy, r, deadline):
    """
    Radial sweep: fix the largest circle as center, then optimize angular positions
    and radial distances of other circles relative to it.
    """
    n = len(xy)
    if n < 2:
        return xy, r, 0.0, False
    
    # Find the largest circle as anchor
    anchor_idx = int(np.argmax(r))
    anchor_x, anchor_y, anchor_r = xy[anchor_idx, 0], xy[anchor_idx, 1], r[anchor_idx]
    
    # Parameters for other circles: angle and radial distance from anchor
    n_other = n - 1
    other_indices = [i for i in range(n) if i != anchor_idx]
    
    # Initial parameters: angle and distance from anchor
    angles = np.zeros(n_other)
    dists = np.zeros(n_other)
    for k, idx in enumerate(other_indices):
        dx = xy[idx, 0] - anchor_x
        dy = xy[idx, 1] - anchor_y
        angles[k] = math.atan2(dy, dx)
        dists[k] = math.sqrt(dx * dx + dy * dy)
    
    radii_other = r[other_indices].copy()
    
    def _radial_objective(params):
        params = np.asarray(params)
        angles = params[0:n_other]
        dists = np.maximum(params[n_other:2*n_other], 0.001)
        radii = np.maximum(params[2*n_other:3*n_other], 0.001)
        
        # Reconstruct positions
        pos = np.zeros((n, 2))
        rad = np.zeros(n)
        pos[anchor_idx] = [anchor_x, anchor_y]
        rad[anchor_idx] = anchor_r
        
        for k, idx in enumerate(other_indices):
            pos[idx, 0] = anchor_x + dists[k] * math.cos(angles[k])
            pos[idx, 1] = anchor_y + dists[k] * math.sin(angles[k])
            rad[idx] = radii[k]
        
        # Check wall constraints
        wall_penalty = 0.0
        for i in range(n):
            x, y, ri = pos[i, 0], pos[i, 1], rad[i]
            for margin in [x - ri, 1.0 - x - ri, y - ri, 1.0 - y - ri]:
                if margin < 0.01:
                    wall_penalty += (0.01 - margin) ** 2
        
        # Check non-overlap
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist + 0.01:
                    gap = dist - min_dist
                    if gap < 0.01:
                        overlap_penalty += (0.01 - gap) ** 2
        
        total_radius = np.sum(rad)
        return -total_radius + wall_penalty + overlap_penalty
    
    # Initial params
    init_params = np.concatenate([angles, dists, radii_other])
    
    bounds = []
    for _ in range(n_other):
        bounds.append((-math.pi, math.pi))
    for _ in range(n_other):
        bounds.append((0.001, 1.0))
    for _ in range(n_other):
        bounds.append((0.001, 0.4))
    
    try:
        result = minimize(
            _radial_objective,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        
        opt_params = result.x
        angles_opt = opt_params[0:n_other]
        dists_opt = np.maximum(opt_params[n_other:2*n_other], 0.001)
        radii_opt = np.maximum(opt_params[2*n_other:3*n_other], 0.001)
        
        pos = np.zeros((n, 2))
        rad = np.zeros(n)
        pos[anchor_idx] = [anchor_x, anchor_y]
        rad[anchor_idx] = anchor_r
        
        for k, idx in enumerate(other_indices):
            pos[idx, 0] = anchor_x + dists_opt[k] * math.cos(angles_opt[k])
            pos[idx, 1] = anchor_y + dists_opt[k] * math.sin(angles_opt[k])
            rad[idx] = radii_opt[k]
        
        total, valid = _eval_packing(pos, rad)
        return pos, rad, total, valid
        
    except Exception:
        return xy, r, 0.0, False


def _hexagonal_init(n):
    """Create a hexagonal close-packed initialization."""
    positions = []
    r = 0.15
    dx = 2 * r
    dy = r * math.sqrt(3)
    
    row = 0
    while len(positions) < n:
        for col in range(6):
            if len(positions) >= n:
                break
            x = r + col * dx + (row % 2) * r
            y = r + row * dy
            if 0 <= x <= 1.0 and 0 <= y <= 1.0:
                positions.append([x, y, r])
        row += 1
    
    # If we have more than n, take the first n
    positions = positions[:n]
    
    # If we have fewer, add some at random
    rng = np.random.default_rng(456)
    while len(positions) < n:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        positions.append([x, y, 0.05])
    
    return np.array(positions)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Strategy 1: Greedy insertion + contact graph optimization
    if time.time() < deadline - 30.0:
        xy = np.empty((0, 2))
        r = np.empty(0)
        
        for i in range(N):
            if time.time() > deadline - 30.0:
                break
            new_xy, new_r, success = _greedy_insertion(xy, r)
            if success:
                xy = new_xy
                r = new_r
            else:
                # Force insert a small circle
                rng = np.random.default_rng(i)
                x = rng.uniform(0.1, 0.9)
                y = rng.uniform(0.1, 0.9)
                ri = 0.02
                xy = np.vstack([xy, [[x, y]]])
                r = np.concatenate([r, [ri]])
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        # Optimize with contact graph
        if time.time() < deadline - 10.0 and best_valid:
            opt_xy, opt_r, opt_total, opt_valid = _contact_graph_optimize(
                best_xy, best_r, deadline - 10.0
            )
            if opt_valid and opt_total > best_total:
                best_total = opt_total
                best_valid = True
                best_xy = opt_xy
                best_r = opt_r
    
    # Strategy 2: Hexagonal init + radial sweep
    if time.time() < deadline - 20.0:
        hex_pos = _hexagonal_init(N)
        xy = hex_pos[:, :2].copy()
        r = hex_pos[:, 2].copy()
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        # Radial sweep optimization
        if time.time() < deadline - 10.0 and best_valid:
            rad_xy, rad_r, rad_total, rad_valid = _radial_sweep_optimize(
                best_xy, best_r, deadline - 10.0
            )
            if rad_valid and rad_total > best_total:
                best_total = rad_total
                best_valid = True
                best_xy = rad_xy
                best_r = rad_r
    
    # Strategy 3: Contact graph optimization from hexagonal init
    if time.time() < deadline - 15.0:
        hex_pos = _hexagonal_init(N)
        xy = hex_pos[:, :2].copy()
        r = hex_pos[:, 2].copy()
        
        opt_xy, opt_r, opt_total, opt_valid = _contact_graph_optimize(
            xy, r, deadline - 15.0
        )
        if opt_valid and opt_total > best_total:
            best_total = opt_total
            best_valid = True
            best_xy = opt_xy
            best_r = opt_r
    
    # Fallback: simple grid
    if best_xy is None or not best_valid:
        side = int(math.ceil(math.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(math.ceil(math.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
