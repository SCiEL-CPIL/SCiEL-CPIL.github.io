"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _largest_empty_circle(xy, r, n_points=2000):
    """
    Find the center and radius of the largest empty circle in the unit square
    given existing circles (xy, r). Uses sampling-based approach for robustness.
    """
    n = len(xy)
    if n == 0:
        return np.array([0.5, 0.5]), 0.5

    # Sample points in the unit square
    rng = np.random.RandomState(42)
    samples = rng.uniform(0.01, 0.99, (n_points, 2))
    
    # Also add centers of existing circles and boundary midpoints
    boundary_pts = np.array([
        [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5],
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [0.25, 0.0], [0.75, 0.0], [0.25, 1.0], [0.75, 1.0],
        [0.0, 0.25], [0.0, 0.75], [1.0, 0.25], [1.0, 0.75],
    ])
    samples = np.vstack([samples, boundary_pts])
    
    best_x, best_y = 0.5, 0.5
    best_r = 0.0

    for px, py in samples:
        # Distance to walls
        wall_dist = min(px, 1.0 - px, py, 1.0 - py)
        
        # Distance to each existing circle center minus its radius
        min_circle_dist = np.inf
        for i in range(n):
            dx = px - xy[i, 0]
            dy = py - xy[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            clearance = dist - r[i]
            if clearance < min_circle_dist:
                min_circle_dist = clearance
        
        r_candidate = min(wall_dist, min_circle_dist)
        if r_candidate > best_r:
            best_r = r_candidate
            best_x, best_y = px, py

    return np.array([best_x, best_y]), best_r


def _local_relaxation(xy, r, deadline, max_iter=500):
    """
    Local relaxation: iteratively adjust each circle's position and radius
    to maximize sum of radii while maintaining disjointness.
    Uses a simple coordinate-descent style approach.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    step = 0.01
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try moving circle i in small steps in all directions
            for dx in [-step, 0.0, step]:
                for dy in [-step, 0.0, step]:
                    new_x = xy[i, 0] + dx
                    new_y = xy[i, 1] + dy
                    
                    # Check if new position is valid with current radius
                    if new_x - r[i] < 0 or 1.0 - new_x - r[i] < 0 or \
                       new_y - r[i] < 0 or 1.0 - new_y - r[i] < 0:
                        continue
                    
                    # Check disjointness with all other circles
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx2 = new_x - xy[j, 0]
                        dy2 = new_y - xy[j, 1]
                        dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Try to increase radius
                    new_r = r[i]
                    max_possible = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
                    for j in range(n):
                        if i == j:
                            continue
                        dx2 = new_x - xy[j, 0]
                        dy2 = new_y - xy[j, 1]
                        dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        max_possible = min(max_possible, dist - r[j])
                    
                    new_r = min(new_r + step * 0.5, max_possible)
                    new_r = max(new_r, 0.001)
                    
                    # Evaluate
                    test_xy = xy.copy()
                    test_r = r.copy()
                    test_xy[i, 0] = new_x
                    test_xy[i, 1] = new_y
                    test_r[i] = new_r
                    
                    total, valid = _eval_packing(test_xy, test_r)
                    if valid and total > best_total + 1e-10:
                        xy = test_xy
                        r = test_r
                        best_total = total
                        best_xy = xy.copy()
                        best_r = r.copy()
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            if no_improve > 5:
                step *= 0.5
                no_improve = 0
                if step < 0.001:
                    break
        else:
            no_improve = 0
    
    return best_xy, best_r, best_total, best_valid


def _sequential_greedy(deadline):
    """
    Sequential greedy insertion: add circles one by one, each at the center
    of the largest empty circle, followed by local relaxation.
    """
    xy_list = []
    r_list = []
    
    # Start with one circle in the center
    xy_list.append(np.array([0.5, 0.5]))
    r_list.append(0.5)
    
    for k in range(1, N):
        if time.time() > deadline:
            break
        
        xy = np.array(xy_list)
        r = np.array(r_list)
        
        # Find largest empty circle
        center, radius = _largest_empty_circle(xy, r, n_points=3000)
        
        # Don't add if radius is too small
        if radius < 0.01:
            # Try with fewer samples or different seed
            center, radius = _largest_empty_circle(xy, r, n_points=5000)
            if radius < 0.01:
                break
        
        xy_list.append(center)
        r_list.append(radius)
    
    if len(xy_list) < N:
        # Fallback: fill remaining with small circles
        while len(xy_list) < N:
            xy = np.array(xy_list)
            r = np.array(r_list)
            center, radius = _largest_empty_circle(xy, r, n_points=2000)
            if radius < 0.005:
                # Place randomly
                rng = np.random.RandomState(42 + len(xy_list))
                center = rng.uniform(0.1, 0.9, 2)
                radius = 0.01
            xy_list.append(center)
            r_list.append(radius)
    
    xy = np.array(xy_list)
    r = np.array(r_list)
    
    # Local relaxation to improve the packing
    best_xy, best_r, best_total, best_valid = _local_relaxation(xy, r, deadline)
    
    return best_xy, best_r, best_total, best_valid


def _hex_init():
    """Hexagonal lattice initialization for 26 circles."""
    positions = []
    # Hexagonal packing in a grid
    row_r = 0.12
    col_spacing = 2 * row_r
    row_spacing = np.sqrt(3) * row_r
    
    row = 0
    while len(positions) < N:
        col = 0
        offset = (row % 2) * col_spacing / 2
        while len(positions) < N:
            x = col_spacing / 2 + col * col_spacing + offset
            y = row_spacing / 2 + row * row_spacing
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append([x, y, row_r])
            col += 1
        row += 1
    
    return np.array(positions[:N])


def _random_init_with_repulsion(n, iterations=100):
    """Random initialization with repulsion to spread circles out."""
    rng = np.random.RandomState(42)
    xy = rng.uniform(0.1, 0.9, (n, 2))
    r = np.full(n, 0.05)
    
    # Repulsion relaxation
    for _ in range(iterations):
        for i in range(n):
            force = np.zeros(2)
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 0.1:
                    force[0] += dx / (dist + 1e-9) * 0.01
                    force[1] += dy / (dist + 1e-9) * 0.01
            
            # Wall repulsion
            if xy[i, 0] < 0.1:
                force[0] += 0.01
            if xy[i, 0] > 0.9:
                force[0] -= 0.01
            if xy[i, 1] < 0.1:
                force[1] += 0.01
            if xy[i, 1] > 0.9:
                force[1] -= 0.01
            
            xy[i] += force
            xy[i] = np.clip(xy[i], 0.05, 0.95)
    
    return xy, r


def _soft_objective(v, n, beta=30.0):
    """
    Soft objective for gradient optimization.
    v = [x_1..x_n, y_1..y_n, r_1..r_n]
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = np.maximum(0.0, r - wall_min)
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)
    
    # Pairwise violations
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = np.maximum(0.0, r_sum - dist)
    iu = np.triu_indices(n, 1)
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol[iu]) / beta)
    
    return -np.sum(r) + 100.0 * (wall_pen + pair_pen)


def _soft_grad(v, n, beta=30.0):
    """Gradient of soft objective."""
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    grad = np.zeros(3 * n)
    
    # Wall penalty
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_sig = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    
    grad[2*n:3*n] += 100.0 * wall_sig
    
    for i in range(n):
        if wall_viol[i] > 0:
            candidates = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_idx = np.argmin(candidates)
            if min_idx == 0:
                grad[i] += -100.0 * wall_sig[i]
            elif min_idx == 1:
                grad[i] += 100.0 * wall_sig[i]
            elif min_idx == 2:
                grad[n + i] += -100.0 * wall_sig[i]
            elif min_idx == 3:
                grad[n + i] += 100.0 * wall_sig[i]
    
    # Pairwise penalty
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dist
    sig = 1.0 / (1.0 + np.exp(-beta * viol))
    iu = np.triu_indices(n, 1)
    sig_iu = sig[iu]
    
    for i in range(n):
        mask = (iu[0] == i) | (iu[1] == i)
        grad[2*n + i] += 100.0 * np.sum(sig_iu[mask])
    
    for i in range(n):
        mask_i = (iu[0] == i)
        mask_j = (iu[1] == i)
        
        if np.any(mask_i):
            js = iu[1][mask_i]
            d = dist[iu[0][mask_i], iu[1][mask_i]]
            d = np.maximum(d, 1e-12)
            grad[i] += 100.0 * np.sum(sig_iu[mask_i] * (x[i] - x[js]) / d)
            grad[n + i] += 100.0 * np.sum(sig_iu[mask_i] * (y[i] - y[js]) / d)
        
        if np.any(mask_j):
            is_ = iu[0][mask_j]
            d = dist[iu[0][mask_j], iu[1][mask_j]]
            d = np.maximum(d, 1e-12)
            grad[i] += 100.0 * np.sum(sig_iu[mask_j] * (x[i] - x[is_]) / d * -1)
            grad[n + i] += 100.0 * np.sum(sig_iu[mask_j] * (y[i] - y[is_]) / d * -1)
    
    grad[2*n:3*n] -= 1.0
    
    return grad


def _gradient_polish(xy, r, deadline, beta=30.0):
    """Gradient-based polish using L-BFGS-B."""
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))
    
    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _soft_objective(v, n, beta)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _soft_grad(v, n, beta)
    
    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9,
                                   'maxfun': 10000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        for i in range(n):
            new_r[i] = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0], new_xy[i, 1], 1.0 - new_xy[i, 1])
            new_r[i] = max(new_r[i], 0.001)
            new_xy[i, 0] = np.clip(new_xy[i, 0], new_r[i], 1.0 - new_r[i])
            new_xy[i, 1] = np.clip(new_xy[i, 1], new_r[i], 1.0 - new_r[i])
        
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Strategy 1: Sequential greedy insertion
    if time.time() < deadline - 60:
        sg_xy, sg_r, sg_total, sg_valid = _sequential_greedy(deadline - 30)
        if sg_valid and sg_total > best_total:
            best_total = sg_total
            best_valid = True
            best_xy = sg_xy
            best_r = sg_r

    # Strategy 2: Hexagonal init + local relaxation
    if time.time() < deadline - 40:
        hex_positions = _hex_init()
        hex_xy = hex_positions[:, :2].copy()
        hex_r = hex_positions[:, 2].copy()
        
        hr_xy, hr_r, hr_total, hr_valid = _local_relaxation(hex_xy, hex_r, deadline - 20)
        if hr_valid and hr_total > best_total:
            best_total = hr_total
            best_valid = True
            best_xy = hr_xy
            best_r = hr_r

    # Strategy 3: Random + repulsion + local relaxation
    if time.time() < deadline - 30:
        rp_xy, rp_r = _random_init_with_repulsion(N)
        rr_xy, rr_r, rr_total, rr_valid = _local_relaxation(rp_xy, rp_r, deadline - 15)
        if rr_valid and rr_total > best_total:
            best_total = rr_total
            best_valid = True
            best_xy = rr_xy
            best_r = rr_r

    # Gradient polish on best result
    if best_valid and best_xy is not None and time.time() < deadline - 20:
        polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline - 5, beta=30.0)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy
            best_r = polished_r

    # Fallback
    if best_xy is None or not best_valid:
        hex_positions = _hex_init()
        best_xy = hex_positions[:, :2].copy()
        best_r = hex_positions[:, 2].copy()
        best_total, best_valid = _eval_packing(best_xy, best_r)
        if not best_valid:
            result = np.zeros((N, 3))
            side = 5
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
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
