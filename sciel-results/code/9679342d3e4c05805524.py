"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_radius(pts):
    """Compute the maximum common radius for the given configuration."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    x = pts[:, 0]
    y = pts[:, 1]
    wall_dists = np.minimum(
        np.minimum(0.5 - x, x + 0.5),
        np.minimum(0.5 - y, y + 0.5)
    )
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _initial_hex_grid(n):
    """Create initial hexagonal grid of n points in [-0.5, 0.5]^2."""
    # Estimate spacing for hex packing
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    pts = []
    y = -0.5 + s / 2
    row = 0
    while y < 0.5 and len(pts) < n:
        x_offset = (row % 2) * s / 2
        x = -0.5 + s / 2 + x_offset
        while x < 0.5 and len(pts) < n:
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                pts.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    
    pts = np.array(pts)
    if len(pts) < n:
        remaining = n - len(pts)
        rand_pts = np.random.rand(remaining, 2) - 0.5
        pts = np.vstack([pts, rand_pts])
    return pts[:n]

def _repulsion_step(pts, target_r, step_size):
    """
    Move points to increase minimum distance toward target_r.
    Returns updated points and whether improvement was made.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Forces from pairs
    forces = np.zeros_like(pts)
    threshold = target_r * 2.0
    
    active = (dists < threshold) & (dists > 1e-10)
    np.fill_diagonal(active, False)
    
    # Weight = 1/d for active pairs
    W = np.zeros_like(dists)
    np.divide(1.0, dists, out=W, where=active)
    
    # grad[i] = sum_j W[i,j] * (pts[i] - pts[j])
    # = sum_j W[i,j]*pts[i] - sum_j W[i,j]*pts[j]
    row_sums = W.sum(axis=1)
    grad_pairs = row_sums[:, np.newaxis] * pts - W.dot(pts)
    
    # Wall forces
    x = pts[:, 0]
    y = pts[:, 1]
    d_right = 0.5 - x
    d_left = x + 0.5
    d_top = 0.5 - y
    d_bottom = y + 0.5
    
    # Only apply wall force if point is within threshold of wall
    wall_active = np.minimum(np.minimum(d_right, d_left), np.minimum(d_top, d_bottom)) < target_r
    
    grad_walls = np.zeros_like(pts)
    for i in range(n):
        if wall_active[i]:
            d_r = d_right[i]
            d_l = d_left[i]
            d_t = d_top[i]
            d_b = d_bottom[i]
            min_d = min(d_r, d_l, d_t, d_b)
            w = 1.0 / max(min_d, 1e-10)
            if d_r == min_d:
                grad_walls[i, 0] -= w
            if d_l == min_d:
                grad_walls[i, 0] += w
            if d_t == min_d:
                grad_walls[i, 1] -= w
            if d_b == min_d:
                grad_walls[i, 1] += w
    
    grad = grad_pairs + grad_walls
    
    new_pts = pts + step_size * grad
    new_pts = np.clip(new_pts, -0.495, 0.495)
    
    return new_pts

def _optimize_repulsion(pts, target_r, deadline, max_steps=5000):
    """
    Run repulsion-based optimization to achieve target_r.
    Returns best points found.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    current_r = best_r
    
    step_size = 0.003
    no_improve = 0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        new_pts = _repulsion_step(pts, target_r, step_size)
        
        # Compute new radius
        new_r = _compute_radius(new_pts)
        
        if new_r > current_r:
            pts = new_pts
            current_r = new_r
            no_improve = 0
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            no_improve += 1
            # Reduce step size if stuck
            if no_improve > 50:
                step_size *= 0.9
                no_improve = 0
            # If step size too small, break
            if step_size < 1e-6:
                break
        
        # If we've achieved target, stop
        if current_r >= target_r * 0.999:
            break
    
    return best_pts, best_r

def _bottleneck_local_opt(pts, deadline):
    """
    Fine-tune by focusing on the bottleneck constraints (closest pairs and nearest walls).
    """
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    current_r = best_r
    
    # Find bottleneck pairs and walls
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    x = pts[:, 0]
    y = pts[:, 1]
    wall_dists = np.minimum(
        np.minimum(0.5 - x, x + 0.5),
        np.minimum(0.5 - y, y + 0.5)
    )
    
    min_pair = np.min(dists)
    min_wall = np.min(wall_dists)
    
    # Identify critical points
    critical = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and dists[i, j] < min_pair * 1.05:
                critical[i] = True
                critical[j] = True
        if wall_dists[i] < min_wall * 1.05:
            critical[i] = True
    
    # Try small perturbations on critical points
    n_crit = np.sum(critical)
    if n_crit == 0:
        return best_pts, best_r
    
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001]
    
    for ss in step_sizes:
        for _ in range(200):
            if time.time() > deadline:
                return best_pts, best_r
            
            # Pick a random critical point
            crit_indices = np.where(critical)[0]
            i = crit_indices[np.random.randint(len(crit_indices))]
            
            # Try random small move
            delta = np.random.randn(2) * ss
            new_pos = pts[i] + delta
            
            # Check bounds
            if new_pos[0] < -0.495 or new_pos[0] > 0.495 or new_pos[1] < -0.495 or new_pos[1] > 0.495:
                continue
            
            # Update and check
            old_pos = pts[i].copy()
            pts[i] = new_pos
            new_r = _compute_radius(pts)
            
            if new_r > current_r:
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
            else:
                pts[i] = old_pos
    
    return best_pts, best_r

def _polish(pts, deadline):
    """
    Final polish using Nelder-Mead on the objective function.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Use scipy's minimize on a transformed objective
    try:
        from scipy.optimize import minimize
    except ImportError:
        return best_pts, best_r
    
    # Flatten points
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 2)
        # Add penalty for going out of bounds
        penalty = 0
        if np.any(p < -0.5) or np.any(p > 0.5):
            penalty = 1.0
        r = _compute_radius(p)
        return -r + penalty
    
    # Try Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'fatol': 1e-8, 'xatol': 1e-8})
        new_pts = result.x.reshape(-1, 2)
        new_pts = np.clip(new_pts, -0.495, 0.495)
        new_r = _compute_radius(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_r

def construct():
    """Place 200 equal circles in a unit square to maximize the minimum radius."""
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s limit
    
    # Start with hexagonal grid
    pts = _initial_hex_grid(N)
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Multi-scale approach: gradually increase target radius
    # Start from current radius and increase in steps
    target_r = current_r * 1.01
    
    # Main optimization loop
    max_iterations = 20
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:
            break
        
        # Try to achieve target_r
        new_pts, new_r = _optimize_repulsion(pts, target_r, deadline - 30, max_steps=2000)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
            current_r = new_r
        
        # Increase target
        target_r = current_r * 1.02
        
        # If we're not making progress, try a different approach
        if new_r <= current_r * 1.001:
            # Add small perturbation and retry
            pts = pts + np.random.randn(N, 2) * 0.002
            pts = np.clip(pts, -0.49, 0.49)
            current_r = _compute_radius(pts)
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
    
    # Bottleneck local optimization
    if time.time() < deadline - 15:
        pts, r = _bottleneck_local_opt(pts, deadline - 15)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        pts, r = _polish(pts, deadline - 5)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final verification
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
    
    return best_pts

# EVOLVE-BLOCK-END
