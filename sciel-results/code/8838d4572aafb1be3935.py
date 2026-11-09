"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the exact maximum common radius for centres c."""
    n = len(c)
    if n == 0:
        return 0.0
    # Boundary clearance
    d_boundary = np.min(0.5 - np.abs(c), axis=1)
    min_boundary = np.min(d_boundary)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_boundary)


def _soft_min_value_and_grad(c, beta, top_k=500):
    """
    Compute soft-min radius and gradient using only top-k contributing terms.
    Returns (r_soft, grad).
    """
    n = len(c)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Boundary distances
    b_dist = 0.5 - np.abs(c)  # shape (n, 3)
    
    # Combine: d_pair[i,j] = dist[i,j]/2, d_bound[i,k] = b_dist[i,k]
    # For efficiency, find top-k smallest values
    d_pair = dist / 2.0  # (n, n)
    d_bound = b_dist  # (n, 3)
    
    # Flatten and find top-k smallest
    # We'll work with the full set but only compute gradient for top-k
    d_all_flat = np.concatenate([d_pair.flatten(), d_bound.flatten()])
    
    # Find indices of top_k smallest
    k = min(top_k, len(d_all_flat))
    # Use partition for speed
    if k < len(d_all_flat):
        idx = np.argpartition(d_all_flat, k)[:k]
    else:
        idx = np.arange(len(d_all_flat))
    
    d_top = d_all_flat[idx]
    
    # Soft min over top-k
    min_d = np.min(d_top)
    shifted = d_top - min_d
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    r_soft = min_d - (1.0 / beta) * np.log(S)
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Determine which are pair vs boundary
    n_pairs = n * n  # including diagonal, but diagonal is inf so won't be in top-k typically
    # Actually, let's be more careful. d_pair has n*n entries, d_bound has n*3 entries
    n_pair_entries = n * n
    
    for ii in range(len(idx)):
        w = weights[ii]
        if w < 1e-15:
            continue
        global_idx = idx[ii]
        if global_idx < n_pair_entries:
            # Pair contribution
            i = global_idx // n
            j = global_idx % n
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-12:
                continue
            direction = (c[i] - c[j]) / d
            grad[i] += w * direction
            grad[j] -= w * direction
        else:
            # Boundary contribution
            bi = global_idx - n_pair_entries
            i = bi // 3
            k_axis = bi % 3
            sign_val = np.sign(c[i, k_axis])
            if sign_val != 0:
                grad[i, k_axis] -= w * sign_val
    
    return r_soft, grad


def _fcc_lattice_init(seed=42):
    """Initialize with FCC lattice structure, taking 100 points."""
    rng = np.random.RandomState(seed)
    
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Scale to fit in [-0.5, 0.5]^3
    # Let's use a grid of FCC cells
    
    m = 5  # 5x5x5 grid of FCC cells = 500 points, take 100
    cell_size = 1.0 / m
    points = []
    
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # 4 FCC points per cell
                base = np.array([
                    (ix + 0.0) * cell_size - 0.5,
                    (iy + 0.0) * cell_size - 0.5,
                    (iz + 0.0) * cell_size - 0.5,
                ])
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5 * cell_size, 0.5 * cell_size]),
                    np.array([0.5 * cell_size, 0, 0.5 * cell_size]),
                    np.array([0.5 * cell_size, 0.5 * cell_size, 0]),
                ]
                for off in offsets:
                    p = base + off
                    # Center the whole thing
                    p -= np.array([0.5 * cell_size * (m - 1) / 2] * 3)
                    # Actually, let's just place in [-0.5, 0.5]
                    p = (base - 0.5 * (m - 1) * cell_size) + off
                    # Simpler: just use the raw coordinates and clip
                    p = base + off
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    
    # Shuffle and take 100
    perm = rng.permutation(len(points))
    c = points[perm[:N]].copy()
    
    # Add small jitter
    c += rng.normal(0, 0.005, c.shape)
    c = np.clip(c, -0.495, 0.495)
    
    return c


def _random_init(seed=42):
    """Random initialization inside the cube."""
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, (N, 3))
    return c


def _optimize_from_init(c_init, time_budget=30.0, beta_start=5.0, beta_end=300.0):
    """
    Optimize from an initial configuration using gradient ascent with soft-min.
    Returns the best configuration found.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start = time.time()
    deadline = start + time_budget
    
    step_size = 0.02
    n_steps_max = 5000
    
    for step in range(n_steps_max):
        if time.time() > deadline - 2.0:
            break
        
        progress = step / n_steps_max
        beta = beta_start * (beta_end / beta_start) ** progress
        
        r, grad = _soft_min_value_and_grad(c, beta, top_k=300)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            # Check actual radius
            actual_r = _compute_radius(c)
            if actual_r > best_r:
                best_r = actual_r
                best_c = c.copy()
            break
        
        # Backtracking line search
        alpha = step_size
        improved = False
        for _ in range(20):
            c_new = c + alpha * grad / grad_norm
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_new, _ = _soft_min_value_and_grad(c_new, beta, top_k=300)
            
            if r_new > r + 1e-10:
                c = c_new
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # Try random perturbation
            c_pert = c + np.random.normal(0, 0.003, c.shape)
            c_pert = np.clip(c_pert, -0.4999, 0.4999)
            r_pert, _ = _soft_min_value_and_grad(c_pert, beta, top_k=300)
            if r_pert > r:
                c = c_pert
            else:
                step_size *= 0.9
                if step_size < 1e-5:
                    break
        
        # Periodically check actual radius
        if step % 50 == 0:
            actual_r = _compute_radius(c)
            if actual_r > best_r:
                best_r = actual_r
                best_c = c.copy()
    
    # Final check
    actual_r = _compute_radius(c)
    if actual_r > best_r:
        best_r = actual_r
        best_c = c.copy()
    
    return best_c, best_r


def _active_set_polish(c, time_budget=5.0):
    """
    Polish the configuration by directly moving points to satisfy the tightest constraints.
    """
    c = c.copy()
    start = time.time()
    deadline = start + time_budget
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Find the tightest constraint
        # Compute all pair distances and boundary distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        b_dist = 0.5 - np.abs(c)
        min_b = np.min(b_dist, axis=1)
        
        # Find the minimum of (dists/2, min_b)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(min_b)
        
        r = min(min_pair, min_bound)
        
        if r > 0.11138:  # Already good
            break
        
        # Find which constraint is tightest
        if min_pair <= min_bound:
            # Pair constraint
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i == j:
                break
            d = dists[i, j]
            if d < 1e-10:
                break
            # Move i and j apart by small amount
            direction = (c[i] - c[j]) / d
            delta = (2 * r - d) / 2 + 1e-4
            if delta > 0:
                c[i] += direction * delta / 2
                c[j] -= direction * delta / 2
        else:
            # Boundary constraint
            idx = np.argmin(min_b)
            axis = np.argmin(b_dist[idx])
            # Move point idx away from boundary
            if c[idx, axis] > 0:
                c[idx, axis] -= (r - min_b[idx]) + 1e-4
            else:
                c[idx, axis] += (r - min_b[idx]) + 1e-4
        
        c = np.clip(c, -0.49999, 0.49999)
    
    return c


def construct():
    start_time = time.time()
    total_deadline = start_time + 145.0  # 5s safety margin
    
    best_c = None
    best_r = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 6
    time_per_restart = (total_deadline - start_time - 10.0) / n_restarts  # Reserve 10s for polish
    
    for restart in range(n_restarts):
        if time.time() > total_deadline - 15.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            c_init = _fcc_lattice_init(seed=42)
        elif restart == 1:
            c_init = _fcc_lattice_init(seed=123)
        elif restart == 2:
            c_init = _random_init(seed=77)
        elif restart == 3:
            c_init = _fcc_lattice_init(seed=999)
        elif restart == 4:
            c_init = _random_init(seed=555)
        else:
            c_init = _fcc_lattice_init(seed=2024)
        
        remaining = total_deadline - time.time() - 15.0
        if remaining < 5.0:
            break
        
        t_budget = min(time_per_restart, remaining)
        c_opt, r_opt = _optimize_from_init(c_init, time_budget=t_budget)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # Print progress
        print(f"Restart {restart}: r = {r_opt:.10f}, best = {best_r:.10f}, time = {time.time()-start_time:.1f}s")
    
    # Final polish on the best configuration
    if best_c is not None and time.time() < total_deadline - 3.0:
        polish_time = min(10.0, total_deadline - time.time() - 2.0)
        best_c = _active_set_polish(best_c, time_budget=polish_time)
        best_r = _compute_radius(best_c)
        print(f"After polish: r = {best_r:.10f}")
    
    # Ensure valid output
    if best_c is None:
        best_c = _fcc_lattice_init(seed=42)
    
    best_c = np.clip(best_c, -0.499999, 0.499999)
    
    return best_c

# EVOLVE-BLOCK-END
