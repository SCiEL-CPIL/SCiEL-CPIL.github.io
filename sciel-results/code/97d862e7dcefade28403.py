"""
Pack 200 equal circles in a unit square using gradient-based optimization
with a soft-min objective, combined with targeted perturbation to escape
local minima.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
_HARD_LIMIT = 145  # Hard wall-clock limit with safety margin (evaluator kills at 150s)

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    """
    if len(pts) == 0:
        return 0.0
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _hard_pgd(pts, time_limit, initial_step=0.001, decay=0.999, min_step=1e-6, beta=1000.0):
    """
    Projected Gradient Ascent for maximizing the minimum radius.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    step = initial_step
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    while time.time() - t_start < time_limit:
        # 1. Compute constraints
        w_left = pts[:, 0] + 0.5
        w_right = 0.5 - pts[:, 0]
        w_bottom = pts[:, 1] + 0.5
        w_top = 0.5 - pts[:, 1]
        
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        C = np.empty(4*N + len(i_idx))
        C[0:N] = w_left
        C[N:2*N] = w_right
        C[2*N:3*N] = w_bottom
        C[3*N:4*N] = w_top
        C[4*N:] = dists / 2.0
        
        r_curr = np.min(C)
        
        if r_curr > best_r:
            best_r = r_curr
            best_pts = pts.copy()
            
        # 2. Compute Gradient of soft-min
        y = -beta * C
        max_y = np.max(y)
        if not np.isfinite(max_y):
            break
            
        shifted = y - max_y
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            break
            
        weights = exp_shifted / sum_exp
        
        grad_pts = np.zeros((N, 2))
        
        # Wall gradients
        grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
        grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
        
        # Pair gradients
        valid = dists > 1e-10
        dx_safe = np.where(valid, dx, 1.0)
        dy_safe = np.where(valid, dy, 1.0)
        dists_safe = np.where(valid, dists, 1.0)
        
        grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
        grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
        grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
        grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
        
        pair_weights = weights[4*N:]
        
        np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
        np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
        np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
        np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
        
        # 3. Update points
        norm = np.linalg.norm(grad_pts, axis=1, keepdims=True)
        norm = np.where(norm > 1e-10, norm, 1.0)
        grad_pts = grad_pts / norm
        
        pts += step * grad_pts
        pts = np.clip(pts, -0.5, 0.5)
        
        # 4. Decay step
        step *= decay
        if step < min_step:
             break

    return best_r, best_pts

def _generate_hex_initial(spacing=None):
    """
    Generate a hexagonal lattice packing.
    """
    if spacing is None:
        spacing = 0.0732
    
    s = spacing
    points = []
    row_h = s * np.sqrt(3) / 2
    
    # Generate a grid larger than the square
    y = -0.6
    row_idx = 0
    while y < 0.6:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.6 + shift
        while x < 0.6:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    # Filter points inside the square
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        # Try smaller spacing
        s = 0.070
        points = []
        row_h = s * np.sqrt(3) / 2
        y = -0.6
        row_idx = 0
        while y < 0.6:
            shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points.append([x, y])
                x += s
            y += row_h
            row_idx += 1
        pts = np.array(points)
        mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
               (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
        pts = pts[mask]
        
    if len(pts) >= N:
        # Select N points closest to the center
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
        return pts
    else:
        return np.random.uniform(-0.4, 0.4, (N, 2))

def _vectorized_fd(pts, time_limit, max_iter=1000):
    """
    Vectorized force-directed repulsion to spread out overlapping circles.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    step = 0.005
    
    for _ in range(max_iter):
        if time.time() - t_start > time_limit:
            break
            
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        threshold = 0.08
        mask = (dists < threshold) & (dists > 1e-6)
        
        mag = np.zeros_like(dists)
        mag[mask] = 1.0 / (dists[mask] ** 2)
        
        force_vec = diff * mag[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Boundary repulsion
        margin = 0.05
        forces[:, 0] += np.clip(-0.5 + margin - pts[:, 0], 0, None) * 10.0
        forces[:, 0] -= np.clip(pts[:, 0] - (0.5 - margin), 0, None) * 10.0
        forces[:, 1] += np.clip(-0.5 + margin - pts[:, 1], 0, None) * 10.0
        forces[:, 1] -= np.clip(pts[:, 1] - (0.5 - margin), 0, None) * 10.0
        
        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)
        
        step *= 0.99
        
        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    return best_r, best_pts

def _perturb_and_optimize(pts, t_start, time_budget, rng):
    """
    Perturb the configuration and run Hard PGD.
    """
    r_curr = _get_radius(pts)
    
    pts_new = pts.copy()
    # Perturb a random subset of points
    num_perturb = 10
    idx = rng.integers(0, N, size=num_perturb)
    pts_new[idx] += rng.normal(0, 0.005, (num_perturb, 2))
    pts_new = np.clip(pts_new, -0.5, 0.5)
    
    r_opt, pts_opt = _hard_pgd(pts_new, time_budget)
    
    if r_opt > r_curr:
        return r_opt, pts_opt
    else:
        return r_curr, pts

def construct():
    t0 = time.time()
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_r = 0.0
    best_pts = None
    
    # Generate initial hexagonal configuration
    pts_hex = _generate_hex_initial(spacing=0.0732)
    
    # Initial vectorized force-directed phase
    fd_time = min(5.0, _HARD_LIMIT * 0.05)
    r_fd, pts_fd = _vectorized_fd(pts_hex, time_limit=fd_time)
    
    if r_fd > best_r:
        best_r = r_fd
        best_pts = pts_fd
    
    # Main optimization loop with perturbations
    # Use Hard PGD
    num_rounds = 10
    time_per_round = (_HARD_LIMIT - 10) / num_rounds
    
    for round_idx in range(num_rounds):
        elapsed = time.time() - t0
        remaining = _HARD_LIMIT - 5 - elapsed
        if remaining < time_per_round:
            break
        
        # Run Hard PGD
        r_opt, pts_opt = _hard_pgd(best_pts, time_limit=time_per_round * 0.7)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt
        
        # Perturb and try to escape local minimum
        elapsed = time.time() - t0
        remaining = _HARD_LIMIT - 5 - elapsed
        if remaining > time_per_round * 0.3:
            r_pert, pts_pert = _perturb_and_optimize(
                best_pts, t0, remaining * 0.8, rng
            )
            if r_pert > best_r:
                best_r = r_pert
                best_pts = pts_pert
    
    # Final polish with smaller steps
    elapsed = time.time() - t0
    remaining = _HARD_LIMIT - 3 - elapsed
    if remaining > 5:
        r_polish, pts_polish = _hard_pgd(best_pts, time_limit=remaining, initial_step=0.0005, decay=0.9995)
        if r_polish > best_r:
            best_r = r_polish
            best_pts = pts_polish
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
