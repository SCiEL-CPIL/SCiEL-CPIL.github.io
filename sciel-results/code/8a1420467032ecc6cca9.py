"""Pack 200 equal circles in a unit square using discrete lattice search + continuous polish."""
import numpy as np
import time
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
        if min_pair_dist == np.inf:
            from scipy.spatial.distance import cdist
            D = cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _generate_hex_lattice(spacing, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal lattice of points inside the square [-0.5, 0.5]^2."""
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing

    y = -0.5 + offset_y
    row_idx = 0
    while y <= 0.5 + 1e-9:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + offset_x + shift
        while x <= 0.5 + 1e-9:
            if -0.5 - 1e-9 <= x <= 0.5 + 1e-9 and -0.5 - 1e-9 <= y <= 0.5 + 1e-9:
                pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1

    if not pts:
        return np.array([])
    return np.array(pts)


def _greedy_select(pts, N, max_time=None):
    """
    From a larger set of points, greedily select N points that maximize the minimum distance.
    Uses a farthest-point heuristic with KDTree acceleration.
    """
    if len(pts) <= N:
        return pts.copy()

    n_total = len(pts)
    selected = []
    available = list(range(n_total))
    
    # Start with a random point
    idx0 = np.random.randint(n_total)
    selected.append(idx0)
    available.remove(idx0)
    
    # Maintain a KDTree of selected points for fast distance queries
    selected_pts = pts[selected]
    
    for k in range(1, N):
        if max_time is not None and time.time() > max_time:
            break
            
        # For each available point, compute min distance to selected
        # Use KDTree query for efficiency
        if len(selected) == 1:
            # Just distance to the single selected point
            diffs = pts[available] - selected_pts[0]
            dists = np.linalg.norm(diffs, axis=1)
        else:
            tree = KDTree(selected_pts)
            dists, _ = tree.query(pts[available], k=1)
        
        # Pick the farthest
        best_local = np.argmax(dists)
        best_global = available[best_local]
        selected.append(best_global)
        available.remove(best_global)
        selected_pts = np.vstack([selected_pts, pts[best_global]])
    
    return pts[selected]


def _local_polish(pts, r_target, max_iters=5000, learning_rate=0.001, cooling=0.9998,
                  time_limit=None):
    """
    Local optimization using gradient descent on the soft-min objective.
    """
    pts = pts.copy()
    n = len(pts)
    current_lr = learning_rate
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Compute forces/gradient
        forces = _compute_forces(pts, r_target)
        
        # Update positions
        pts += current_lr * forces
        
        # Project back to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], 0.5)
        
        # Cool down
        current_lr *= cooling
        
        # Early stopping if forces are very small
        if np.max(np.abs(forces)) < 1e-7:
            break
    
    return pts


def _compute_forces(pts, r_target, repulsion_strength=10.0, wall_strength=20.0):
    """
    Compute repulsive forces between circles and wall forces.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Build KDTree for neighbor queries
    tree = KDTree(pts)
    cutoff = max(2 * r_target * 1.5, 0.1)
    
    for i in range(n):
        # Wall forces
        x, y = pts[i, 0], pts[i, 1]
        
        if x < -0.5 + r_target:
            overlap = r_target - (x + 0.5)
            forces[i, 0] += wall_strength * overlap
        if x > 0.5 - r_target:
            overlap = x - (0.5 - r_target)
            forces[i, 0] -= wall_strength * overlap
        if y < -0.5 + r_target:
            overlap = r_target - (y + 0.5)
            forces[i, 1] += wall_strength * overlap
        if y > 0.5 - r_target:
            overlap = y - (0.5 - r_target)
            forces[i, 1] -= wall_strength * overlap
        
        # Repulsive forces from neighbors
        neighbors = tree.query_ball_point(pts[i], cutoff)
        for j in neighbors:
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                forces[i] += np.random.randn(2) * 0.01
                continue
            if dist < 2 * r_target:
                overlap = 2 * r_target - dist
                force_mag = repulsion_strength * overlap / (dist * dist + 1e-10)
                forces[i] += force_mag * diff
    
    return forces


def _soft_min_objective(pts, beta=100.0):
    """
    Compute soft-min objective for smooth optimization.
    """
    n = len(pts)
    
    # Wall distances
    wall_dists = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pairwise distances / 2
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        pair_dists = np.array([np.inf])
    else:
        pair_dists = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in pairs])
    
    all_vals = np.concatenate([wall_dists, pair_dists])
    
    # Soft min using log-sum-exp
    neg_vals = -np.asarray(all_vals)
    max_neg = np.max(neg_vals)
    shifted = neg_vals - max_neg
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_neg
    return -log_sum_exp


def _smooth_polish(pts, max_iters=3000, learning_rate=0.0005, cooling=0.9997,
                   beta_start=20.0, beta_end=200.0, time_limit=None):
    """
    Gradient-based optimization using soft-min objective.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Interpolate beta
        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute objective and gradient
        obj, grad = _objective_and_grad(pts, beta)
        
        # Update
        pts += learning_rate * grad
        
        # Project to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Cool learning rate
        learning_rate *= cooling
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-7:
            break
    
    return pts


def _objective_and_grad(pts, beta=50.0):
    """
    Compute soft-min objective and its gradient.
    """
    n = len(pts)
    if n == 0:
        return 0.0, np.zeros_like(pts)
    
    # Wall constraints
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair constraints
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    
    if len(pairs) == 0:
        pair_vals = np.array([10.0])
    else:
        pair_vals = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in pairs])
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # For numerical stability
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    soft_min = -(max_neg + np.log(sum_exp)) / beta
    
    grad = np.zeros_like(pts)
    
    # Wall gradients
    weights_wall = exp_shifted[:4*n] / sum_exp
    for i in range(n):
        grad[i, 0] += weights_wall[4*i] * 1.0
        grad[i, 0] += weights_wall[4*i+1] * (-1.0)
        grad[i, 1] += weights_wall[4*i+2] * 1.0
        grad[i, 1] += weights_wall[4*i+3] * (-1.0)
    
    # Pair gradients
    if len(pairs) > 0:
        weights_pair = exp_shifted[4*n:] / sum_exp
        for k, (i, j) in enumerate(pairs):
            if k >= len(pairs):
                break
            w = weights_pair[k]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad[i] += w * diff / (2.0 * dist)
            grad[j] -= w * diff / (2.0 * dist)
    
    return soft_min, grad


def construct():
    t0 = time.time()
    
    best_r = 0.0
    best_pts = None
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Generate hexagonal lattices with various spacings and offsets
    # 2. Greedily select N points from each lattice
    # 3. Polish with local optimization
    # 4. Keep the best result
    
    # Try different spacings and offsets
    spacings = [0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09]
    offsets = []
    for s in spacings:
        for _ in range(4):
            off_x = np.random.uniform(-s/2, s/2)
            off_y = np.random.uniform(-s/2, s/2)
            offsets.append((s, off_x, off_y))
    
    # Shuffle
    np.random.shuffle(offsets)
    
    for spacing, off_x, off_y in offsets:
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
        
        # Generate hex lattice
        lattice = _generate_hex_lattice(spacing, off_x, off_y)
        
        if len(lattice) < N:
            continue
        
        # Greedily select N points
        selected = _greedy_select(lattice, N, max_time=_TIME_LIMIT - 5)
        
        if len(selected) < N:
            continue
        
        # Compute initial radius
        r_init = _get_radius(selected)
        
        # Polish with local optimization
        polished = _local_polish(selected, r_init * 1.02, max_iters=3000,
                                 learning_rate=0.001, cooling=0.9998,
                                 time_limit=_TIME_LIMIT - 5)
        r_polished = _get_radius(polished)
        
        # Also try smooth polish
        if time.time() - t0 < _TIME_LIMIT - 5:
            smooth_pts = _smooth_polish(polished, max_iters=2000, learning_rate=0.0005,
                                        cooling=0.9997, beta_start=20.0, beta_end=200.0,
                                        time_limit=_TIME_LIMIT - 2)
            r_smooth = _get_radius(smooth_pts)
            if r_smooth > r_polished:
                r_polished = r_smooth
                polished = smooth_pts
        
        r_best = max(r_polished, r_init)
        pts_best = polished if r_polished >= r_init else selected
        
        if r_best > best_r:
            best_r = r_best
            best_pts = pts_best.copy()
    
    # Final polish if we have a good starting point
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        final_polish = _smooth_polish(best_pts, max_iters=2000, learning_rate=0.0005,
                                      cooling=0.9997, beta_start=10.0, beta_end=150.0,
                                      time_limit=_TIME_LIMIT - 2)
        r_final = _get_radius(final_polish)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_polish.copy()
    
    # Fallback if something went wrong
    if best_pts is None:
        # Use a simple grid
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
