"""Pack 200 equal circles in a unit square using rigid-body dynamics with Verlet integration."""
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


def _dynamics_step(pts, r_target, dt=0.002, repulsion_strength=50.0, wall_strength=100.0, damping=0.95):
    """
    One Verlet integration step for rigid-body dynamics.
    - Inverse-square repulsive forces between circles closer than 2*r_target
    - Strong repulsive potential from walls
    - Velocity damping to help convergence
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Build KDTree for neighbor queries
    tree = KDTree(pts)
    cutoff = max(2 * r_target * 1.5, 0.1)
    
    for i in range(n):
        # Wall forces: strong repulsive potential
        # Use a quadratic potential near walls
        x, y = pts[i, 0], pts[i, 1]
        
        # Left wall
        if x < -0.5 + r_target:
            overlap = r_target - (x + 0.5)
            forces[i, 0] += wall_strength * overlap
        # Right wall
        if x > 0.5 - r_target:
            overlap = x - (0.5 - r_target)
            forces[i, 0] -= wall_strength * overlap
        # Bottom wall
        if y < -0.5 + r_target:
            overlap = r_target - (y + 0.5)
            forces[i, 1] += wall_strength * overlap
        # Top wall
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
                # Random nudge for coincident points
                forces[i] += np.random.randn(2) * 0.01
                continue
            if dist < 2 * r_target:
                # Inverse-square repulsion: stronger when closer
                overlap = 2 * r_target - dist
                # Force magnitude proportional to overlap / dist^2 for strong repulsion
                force_mag = repulsion_strength * overlap / (dist * dist + 1e-10)
                forces[i] += force_mag * diff
    
    return forces


def _simulate_dynamics(pts, r_target, max_steps=50000, dt=0.002, damping=0.95, 
                       repulsion_strength=50.0, wall_strength=100.0,
                       time_limit=130.0):
    """
    Run rigid-body dynamics simulation.
    Uses Verlet integration with velocity damping.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Initialize velocities to zero
    velocities = np.zeros_like(pts)
    
    # Track best configuration found
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    for step in range(max_steps):
        # Check time
        if time.time() > time_limit:
            break
        
        # Compute forces
        forces = _dynamics_step(pts, r_target, dt, repulsion_strength, wall_strength, damping)
        
        # Verlet integration: x_new = 2*x - x_old + a*dt^2
        # With damping: v_new = damping * v + a*dt
        velocities = damping * velocities + forces * dt
        pts += velocities * dt
        
        # Project back to square (hard constraint)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Check radius periodically
        if step % 100 == 0:
            r = _get_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Early stopping if velocities are very small
        if np.max(np.abs(velocities)) < 1e-8:
            break
    
    return best_pts, best_r


def _local_polish(pts, r_target, max_iters=2000, learning_rate=0.0005, cooling=0.9995):
    """
    Local optimization pass using gradient-based method to polish the configuration.
    """
    pts = pts.copy()
    current_lr = learning_rate
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        # Compute forces (same as dynamics)
        forces = _dynamics_step(pts, r_target, dt=1.0, repulsion_strength=10.0, wall_strength=20.0)
        
        # Update positions
        pts += current_lr * forces
        
        # Project back to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Cool down
        current_lr *= cooling
        
        # Early stopping
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return pts


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


def _gradient_polish(pts, max_iters=3000, learning_rate=0.0003, cooling=0.9997, 
                     beta_start=20.0, beta_end=200.0):
    """
    Gradient-based optimization using soft-min objective.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 5:
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


def _jittered_grid_init(n_rows, n_cols, jitter=0.005):
    """Generate a jittered grid of points inside the square."""
    dx = 1.0 / n_cols
    dy = 1.0 / n_rows
    xs = -0.5 + (np.arange(n_cols) + 0.5) * dx
    ys = -0.5 + (np.arange(n_rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    
    # Add random jitter
    jitter = np.random.uniform(-jitter, jitter, size=grid.shape)
    grid = grid + jitter
    
    # Clip to square
    grid[:, 0] = np.clip(grid[:, 0], -0.5, 0.5)
    grid[:, 1] = np.clip(grid[:, 1], -0.5, 0.5)
    
    return grid


def _hex_grid_init(spacing=0.08, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal grid of points inside the square."""
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing
    
    y = -0.5 + offset_y
    row_idx = 0
    while y <= 0.5:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + offset_x + shift
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1
        
    if not pts:
        return np.array([])
    return np.array(pts)


def _select_best_subset(pts, N):
    """
    From a larger set of points, select N points that maximize the minimum distance.
    Use greedy selection.
    """
    if len(pts) <= N:
        return pts
    
    selected = []
    available = set(range(len(pts)))
    
    for _ in range(N):
        if not available:
            break
        
        best_idx = None
        best_min_dist = -1
        
        for idx in available:
            if not selected:
                min_dist = np.inf
            else:
                min_dist = np.min([np.linalg.norm(pts[idx] - pts[s]) for s in selected])
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        selected.append(best_idx)
        available.discard(best_idx)
    
    return pts[selected]


def construct():
    t0 = time.time()
    
    best_r = 0.0
    best_pts = None
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Generate several initial configurations (jittered grids, hex grids, random)
    # 2. For each, run rigid-body dynamics simulation
    # 3. Polish with local optimization
    # 4. Keep the best result
    
    # Different initial configurations to try
    configs = []
    
    # Jittered grids with various parameters
    for rows in [10, 12, 14, 15]:
        for cols in [10, 12, 14, 15]:
            if rows * cols >= N:
                configs.append(('grid', rows, cols, 0.005))
    
    # Hex grids with various parameters
    for spacing in [0.06, 0.07, 0.08, 0.09, 0.10]:
        for _ in range(3):
            off_x = np.random.uniform(-spacing/2, spacing/2)
            off_y = np.random.uniform(-spacing/2, spacing/2)
            configs.append(('hex', spacing, off_x, off_y))
    
    # Random configurations
    for _ in range(5):
        configs.append(('random', None, None, None))
    
    # Shuffle configs
    np.random.shuffle(configs)
    
    for config in configs:
        if time.time() - t0 > _TIME_LIMIT - 15:
            break
        
        cfg_type = config[0]
        
        if cfg_type == 'grid':
            rows = config[1]
            cols = config[2]
            jitter = config[3]
            pts = _jittered_grid_init(rows, cols, jitter)
        elif cfg_type == 'hex':
            spacing = config[1]
            off_x = config[2]
            off_y = config[3]
            pts = _hex_grid_init(spacing, off_x, off_y)
        elif cfg_type == 'random':
            pts = np.random.uniform(-0.5, 0.5, size=(N, 2))
        else:
            continue
        
        if len(pts) < N:
            continue
        
        # If we have more than N points, select the best N
        if len(pts) > N:
            pts = _select_best_subset(pts, N)
        
        # Compute initial radius
        r_init = _get_radius(pts)
        
        # Try to improve with dynamics
        # Target radius slightly above current
        r_target = r_init * 1.05
        
        # Run dynamics simulation
        simulated_pts, r_sim = _simulate_dynamics(pts, r_target, max_steps=30000, dt=0.002, 
                                                   damping=0.95, repulsion_strength=50.0, 
                                                   wall_strength=100.0, time_limit=_TIME_LIMIT - 10)
        
        # Polish with local optimization
        polished_pts = _local_polish(simulated_pts, r_sim, max_iters=1500, learning_rate=0.0005, cooling=0.9995)
        r_polished = _get_radius(polished_pts)
        
        # Also try gradient-based polish
        if time.time() - t0 < _TIME_LIMIT - 10:
            grad_pts = _gradient_polish(polished_pts, max_iters=2000, learning_rate=0.0003, cooling=0.9997,
                                        beta_start=20.0, beta_end=200.0)
            r_grad = _get_radius(grad_pts)
            if r_grad > r_polished:
                r_polished = r_grad
                polished_pts = grad_pts
        
        # Compare with original
        r_best = max(r_polished, r_init)
        pts_best = polished_pts if r_polished >= r_init else pts
        
        if r_best > best_r:
            best_r = r_best
            best_pts = pts_best.copy()
        
        # If we're doing well, try more iterations
        if best_r > 0.036 and time.time() - t0 < _TIME_LIMIT - 10:
            r_target2 = best_r * 1.02
            simulated_pts2, r_sim2 = _simulate_dynamics(best_pts, r_target2, max_steps=20000, dt=0.002,
                                                         damping=0.95, repulsion_strength=50.0,
                                                         wall_strength=100.0, time_limit=_TIME_LIMIT - 5)
            polished_pts2 = _local_polish(simulated_pts2, r_sim2, max_iters=1000, learning_rate=0.0003, cooling=0.9995)
            r_polished2 = _get_radius(polished_pts2)
            if r_polished2 > best_r:
                best_r = r_polished2
                best_pts = polished_pts2.copy()
    
    # Final polish with smooth optimization if we have a good starting point
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 15:
        polished_pts = _gradient_polish(best_pts, max_iters=1500, learning_rate=0.0003, cooling=0.9997,
                                        beta_start=10.0, beta_end=150.0)
        r_polished = _get_radius(polished_pts)
        if r_polished > best_r:
            best_r = r_polished
            best_pts = polished_pts.copy()
        
        # Try another round of dynamics from the polished result
        if time.time() - t0 < _TIME_LIMIT - 10:
            r_target3 = best_r * 1.01
            simulated_pts3, r_sim3 = _simulate_dynamics(best_pts, r_target3, max_steps=10000, dt=0.002,
                                                         damping=0.95, repulsion_strength=50.0,
                                                         wall_strength=100.0, time_limit=_TIME_LIMIT - 5)
            polished_pts3 = _local_polish(simulated_pts3, r_sim3, max_iters=500, learning_rate=0.0002, cooling=0.9995)
            r_polished3 = _get_radius(polished_pts3)
            if r_polished3 > best_r:
                best_r = r_polished3
                best_pts = polished_pts3.copy()
    
    # Fallback if something went wrong
    if best_pts is None:
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
