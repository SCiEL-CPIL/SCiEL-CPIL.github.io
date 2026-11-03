"""
Pack 200 equal circles in a unit square using high-beta soft-min L-BFGS-B
and targeted active-set local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s kill

def _compute_radius_fast(pts):
    """
    Fast radius computation using cdist.
    """
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Compute the soft-min objective (negative radius) and its gradient.
    x is a flat array of shape (2*N,).
    Returns (f, grad) where f = -r_soft and grad = df/dx.
    """
    pts = x.reshape(N, 2)
    
    # 1. Compute all constraint values C
    # Wall terms: 4*N
    # Pair terms: N*(N-1)/2
    
    # Wall values
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair values
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Vectorized distance calculation for pairs
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    pair_vals = dists / 2.0
    
    # Combine all values
    C = np.concatenate([wall_vals, pair_vals])
    
    # 2. Compute soft min and weights
    # soft_min(C) = -1/beta * log(sum(exp(-beta * C)))
    # For numerical stability, shift by min(C)
    min_C = np.min(C)
    # If min_C is very negative, exp(-beta * (C - min_C)) might overflow?
    # Actually, C - min_C >= 0. So -beta * (C - min_C) <= 0. exp is safe (<= 1).
    # Wait, the formula is -1/beta * log(sum(exp(-beta * C)))
    # Let's use the stable log-sum-exp form:
    # log(sum(exp(y))) = max(y) + log(sum(exp(y - max(y))))
    # Here y = -beta * C.
    # max(y) = -beta * min(C).
    # y - max(y) = -beta * C - (-beta * min(C)) = -beta * (C - min(C)).
    # Since C >= min(C), (C - min(C)) >= 0, so -beta * (C - min(C)) <= 0.
    # exp of non-positive is <= 1, no overflow.
    
    y_shifted = -beta * (C - min_C)
    sum_exp = np.sum(np.exp(y_shifted))
    
    if sum_exp == 0:
        # Fallback
        return -min_C, np.zeros_like(x)
        
    soft_min_val = min_C - np.log(sum_exp) / beta
    obj_val = -soft_min_val
    
    # Weights w_i = exp(y_i) / sum(exp(y_j)) = exp(y_shifted_i) / sum_exp
    weights = np.exp(y_shifted) / sum_exp
    
    # 3. Compute Gradient
    # grad_obj = - (1/beta) * sum( w_i * grad_C_i )
    # Note: The gradient of the soft min is a weighted average of the gradients of the individual constraints.
    # d(soft_min)/dx = sum( w_i * (dC_i/dx) )
    # So d(obj)/dx = - d(soft_min)/dx = - sum( w_i * (dC_i/dx) )
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # C[0:N] = x_0 + 0.5 -> grad_x = 1
    # C[N:2*N] = 0.5 - x_0 -> grad_x = -1
    # C[2*N:3*N] = x_1 + 0.5 -> grad_y = 1
    # C[3*N:4*N] = 0.5 - x_1 -> grad_y = -1
    
    w_x_left = weights[0:N]
    w_x_right = weights[N:2*N]
    w_y_bottom = weights[2*N:3*N]
    w_y_top = weights[3*N:4*N]
    
    # Accumulate into grad (flat array)
    # x indices: 0, 2, 4...
    # y indices: 1, 3, 5...
    
    # Using np.add.at is safe but might be slow. 
    # Since we have exactly one contribution per point per wall type, we can just index directly.
    # grad[2*k] += w_x_left[k] - w_x_right[k]
    # grad[2*k+1] += w_y_bottom[k] - w_y_top[k]
    
    grad[0::2] += w_x_left - w_x_right
    grad[1::2] += w_y_bottom - w_y_top
    
    # Pair gradients
    # C[4*N + k] = dist(i, j) / 2
    # grad_x_i = dx / (2 * dist)
    # grad_x_j = -dx / (2 * dist)
    # grad_y_i = dy / (2 * dist)
    # grad_y_j = -dy / (2 * dist)
    
    # We need to accumulate these into the main grad array.
    # np.add.at is the standard way to handle repeated indices.
    
    # Calculate the gradient contributions for each pair
    valid = dists > 1e-10
    # For invalid (dist=0), gradient is undefined, but weight should be handled or ignored.
    # If dist is 0, the term is 0, but gradient is problematic.
    # We'll set the contribution to 0 for invalid pairs.
    
    grad_pair_x_i = np.where(valid, dx / (2.0 * dists), 0.0)
    grad_pair_x_j = np.where(valid, -dx / (2.0 * dists), 0.0)
    grad_pair_y_i = np.where(valid, dy / (2.0 * dists), 0.0)
    grad_pair_y_j = np.where(valid, -dy / (2.0 * dists), 0.0)
    
    # Multiply by weights
    w_pairs = weights[4*N:]
    
    # Accumulate
    np.add.at(grad, i_idx * 2, w_pairs * grad_pair_x_i)
    np.add.at(grad, j_idx * 2, w_pairs * grad_pair_x_j)
    np.add.at(grad, i_idx * 2 + 1, w_pairs * grad_pair_y_i)
    np.add.at(grad, j_idx * 2 + 1, w_pairs * grad_pair_y_j)
    
    # The objective is -soft_min, so we negate the gradient
    grad *= -1
    
    return obj_val, grad

def _hexagonal_initialization(n, jitter=0.0):
    """
    Generate a hexagonal layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    if jitter > 0:
        points = points + jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)
    
    return points

def _grid_initialization(n, jitter=0.0):
    """
    Generate a grid layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) < n:
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])
    
    grid = grid[:n]
    
    if jitter > 0:
        grid = grid + jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid

def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def _force_directed_refine(pts, steps=100, deadline=None):
    """
    Force-directed refinement to untangle initial layout.
    """
    if deadline is None:
        deadline = time.time() + 5
        
    pts = pts.copy()
    N = len(pts)
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Compute forces
        # Vectorized pairwise forces
        vecs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(vecs, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        # Repulsion: F ~ 1/d^2
        # We want to push apart.
        # Direction is vecs / dists.
        # Magnitude is 1 / dists^2.
        
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        unit_vecs = vecs / dists_safe[:, :, np.newaxis]
        
        # Force magnitude
        # Use a softer potential to avoid explosion? 
        # 1/d^2 is standard for Coulomb.
        force_mags = 1.0 / (dists_safe**2)
        force_mags[np.arange(N), np.arange(N)] = 0.0
        
        # Accumulate forces
        forces = np.sum(unit_vecs * force_mags[:, :, np.newaxis], axis=1)
        
        # Boundary forces (soft walls)
        # Push back if close to boundary
        margin = 0.05
        # x
        left_viol = np.maximum(0, (-0.5 + margin) - pts[:, 0])
        right_viol = np.maximum(0, pts[:, 0] - (0.5 - margin))
        forces[:, 0] += 10.0 * (left_viol - right_viol)
        
        # y
        bot_viol = np.maximum(0, (-0.5 + margin) - pts[:, 1])
        top_viol = np.maximum(0, pts[:, 1] - (0.5 - margin))
        forces[:, 1] += 10.0 * (bot_viol - top_viol)
        
        # Update positions
        # Cooling step size
        step_size = 0.05 * (1.0 - step / steps)
        pts += forces * step_size
        
        # Clip to square
        pts = np.clip(pts, -0.5, 0.5)
        
    return pts

def _active_set_polish(pts, max_iters=200, deadline=None):
    """
    Targeted local search on active constraints (bottlenecks).
    """
    if deadline is None:
        deadline = time.time() + 10
        
    pts = pts.copy()
    r = _compute_radius_fast(pts)
    best_r = r
    best_pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair_dist = np.min(D)
        min_pair_idx = np.unravel_index(np.argmin(D), D.shape)
        
        min_wall_dist = np.min([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1])
        ])
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
            
        if min_pair_dist / 2.0 <= min_wall_dist:
            i, j = min_pair_idx
            if i > j: i, j = j, i
            
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-10:
                angle = np.random.uniform(0, 2*np.pi)
                dx, dy = np.cos(angle), np.sin(angle)
                dist = 1.0
            
            step = 0.001 * np.random.uniform(0.5, 1.5)
            
            new_pts = pts.copy()
            new_pts[i, 0] += (dx / dist) * step
            new_pts[i, 1] += (dy / dist) * step
            new_pts[j, 0] -= (dx / dist) * step
            new_pts[j, 1] -= (dy / dist) * step
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
        else:
            min_x = np.min(pts[:, 0]) + 0.5
            max_x = 0.5 - np.max(pts[:, 0])
            min_y = np.min(pts[:, 1]) + 0.5
            max_y = 0.5 - np.max(pts[:, 1])
            
            walls = [min_x, max_x, min_y, max_y]
            wall_idx = np.argmin(walls)
            
            if wall_idx == 0:
                idx = np.argmin(pts[:, 0])
                axis = 0
                direction = 1
            elif wall_idx == 1:
                idx = np.argmax(pts[:, 0])
                axis = 0
                direction = -1
            elif wall_idx == 2:
                idx = np.argmin(pts[:, 1])
                axis = 1
                direction = 1
            else:
                idx = np.argmax(pts[:, 1])
                axis = 1
                direction = -1
            
            step = 0.001 * np.random.uniform(0.5, 1.5)
            
            new_pts = pts.copy()
            if axis == 0:
                new_pts[idx, 0] += direction * step
            else:
                new_pts[idx, 1] += direction * step
            
            perp_axis = 1 - axis
            new_pts[idx, perp_axis] += np.random.uniform(-0.001, 0.001)
            new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius_fast(new_pts)
        
        if new_r >= current_r - 1e-9:
            pts = new_pts
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Prioritize hex and grid with jitter
    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.005)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.005)),
        ("random", lambda: _random_initialization(N)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    for name, init_func in initializers:
        if time.time() > deadline - 30:
            break
        
        try:
            coords = init_func()
            
            # Force-directed refinement
            if time.time() < deadline - 25:
                fd_deadline = time.time() + 5
                if time.time() + 5 > deadline - 25:
                    fd_deadline = deadline - 25
                coords = _force_directed_refine(coords, steps=100, deadline=fd_deadline)
            
            r_init = _compute_radius_fast(coords)
            
            if r_init > best_radius:
                best_radius = r_init
                best_coords = coords.copy()
            
            # Run L-BFGS optimization
            x0 = coords.flatten()
            beta = 100.0  # Higher beta for sharper min
            
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            # Allocate time for optimization
            opt_deadline = time.time() + 10
            if opt_deadline > deadline - 10:
                opt_deadline = deadline - 10
            
            # L-BFGS-B doesn't have a direct time limit, but we can limit iterations
            # or just let it run and check time after.
            # 200 iterations is usually fast enough.
            
            res = minimize(
                _soft_min_objective_and_grad,
                x0,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < -1e-10:
                opt_coords = res.x.reshape(N, 2)
                opt_r = _compute_radius_fast(opt_coords)
                
                if opt_r > best_radius:
                    best_radius = opt_r
                    best_coords = opt_coords.copy()
            
            # Run active set polish on the best so far from this initialization
            if time.time() < deadline - 5:
                polish_deadline = time.time() + 5
                if polish_deadline > deadline - 2:
                    polish_deadline = deadline - 2
                polished_coords, polished_r = _active_set_polish(best_coords, max_iters=200, deadline=polish_deadline)
                if polished_r > best_radius:
                    best_radius = polished_r
                    best_coords = polished_coords.copy()
                    
        except Exception as e:
            continue
    
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)
        
    return best_coords

# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)

def objective(coords):
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    return _hexagonal_initialization(n)

# EVOLVE-BLOCK-END
