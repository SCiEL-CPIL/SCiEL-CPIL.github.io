import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin from 150s

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    if np.any(np.abs(p) > 0.5 + 1e-12):
        return 0.0
    
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint
    dists = cdist(p, p, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _softmin_radius_and_grad(p, beta=50.0):
    """
    Smooth approximation of min radius using softmin, with analytic gradient.
    Returns (softmin_value, gradient_wrt_p) where gradient is d(softmin)/dp.
    """
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)
    b = 0.5 - max_abs
    
    dists = cdist(p, p, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pd = dists / 2.0
    
    iu = np.triu_indices(N, k=1)
    pd_upper = pd[iu]
    
    all_constraints = np.concatenate([b, pd_upper])
    
    # Stable soft-min
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    softmin_val = min_val - log_sum / beta
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(p)
    
    # Boundary part
    boundary_weights = weights[:N]
    max_axes = np.argmax(abs_p, axis=1)
    for j in range(N):
        w = boundary_weights[j]
        if w < 1e-15:
            continue
        k = max_axes[j]
        grad[j, k] += w * (-np.sign(p[j, k]))
    
    # Pairwise part
    pair_weights = weights[N:]
    if np.any(pair_weights > 1e-15):
        i_idx = iu[0]
        j_idx = iu[1]
        diffs = p[i_idx] - p[j_idx]
        dists_pair = np.linalg.norm(diffs, axis=1)
        safe_dists = np.where(dists_pair > 1e-12, dists_pair, 1.0)
        unit_vecs = diffs / safe_dists[:, np.newaxis]
        
        w_scaled = pair_weights * 0.5
        np.add.at(grad, i_idx, w_scaled[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -w_scaled[:, np.newaxis] * unit_vecs)
    
    return softmin_val, grad

def _smooth_obj_and_grad(p, beta=50.0):
    """Objective to maximize: negative softmin radius, with gradient."""
    val, grad = _softmin_radius_and_grad(p, beta)
    return -val, -grad

def _vectorized_repulsion_anneal(pts, n_steps=100, lr_start=0.02, lr_end=0.002, time_limit=None, start_time=None):
    """
    Vectorized repulsion annealing: iteratively push spheres apart.
    Uses 1/r^3 force law for pairwise repulsion and wall repulsion.
    """
    pts = pts.copy().astype(np.float64)
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        # Compute forces (fully vectorized)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1e6)
        
        # Force: diff / dist^3 (repulsive)
        inv_dist3 = 1.0 / (dists ** 3 + 1e-8)
        np.fill_diagonal(inv_dist3, 0)
        
        forces += np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        for k in range(3):
            pos = pts[:, k]
            wall_dist_pos = 0.5 - pos
            wall_dist_neg = 0.5 + pos
            
            force_pos = np.where(wall_dist_pos < 0.15, -1.0 / (wall_dist_pos + 0.01), 0.0)
            force_neg = np.where(wall_dist_neg < 0.15, 1.0 / (wall_dist_neg + 0.01), 0.0)
            
            forces[:, k] += force_pos + force_neg
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def _metropolis_mc(p0, n_steps=20000, time_limit=None, start_time=None):
    """
    Metropolis Monte Carlo to escape local minima with incremental updates.
    """
    p = p0.copy()
    
    D = cdist(p, p, metric='euclidean')
    np.fill_diagonal(D, np.inf)
    b = 0.5 - np.max(np.abs(p), axis=1)
    
    current_r = min(np.min(b), np.min(D) / 2.0)
    best_r = current_r
    best_p = p.copy()
    
    min_b_val = np.min(b)
    min_d_val = np.min(D) / 2.0
    
    step_size = 0.01
    T = 0.01
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        i = np.random.randint(N)
        axis = np.random.randint(3)
        delta = np.random.normal(0, step_size)
        
        old_val = p[i, axis]
        new_val = old_val + delta
        
        if new_val < -0.5:
            new_val = -0.5
        elif new_val > 0.5:
            new_val = 0.5
            
        if new_val == old_val:
            continue

        old_b_i = b[i]
        other_abs = np.abs(np.delete(p[i], axis))
        new_abs_i = np.max(np.concatenate([other_abs, [abs(new_val)]]))
        new_b_i = 0.5 - new_abs_i
        
        p_old_i = p[i].copy()
        p_new_i = p_old_i.copy()
        p_new_i[axis] = new_val
        
        diffs = p - p_new_i
        new_dists_i = np.linalg.norm(diffs, axis=1)
        new_dists_i[i] = np.inf
        
        D[i, :] = new_dists_i
        D[:, i] = new_dists_i
        b[i] = new_b_i
        
        if new_b_i < min_b_val:
            min_b_val_new = new_b_i
        elif old_b_i == min_b_val:
            min_b_val_new = np.min(b)
        else:
            min_b_val_new = min_b_val
        
        min_d_val_new = np.min(D) / 2.0
        
        min_b_val = min_b_val_new
        min_d_val = min_d_val_new
        
        r_trial = min(min_b_val, min_d_val)
        
        if r_trial >= current_r or np.random.random() < np.exp((current_r - r_trial) / T):
            p[i, axis] = new_val
            current_r = r_trial
            if current_r > best_r:
                best_r = current_r
                best_p = p.copy()
        
        T *= 0.999
        step_size *= 0.9999
    
    return best_p, best_r

def _lloyd_step(p, n_steps=1, time_limit=None, start_time=None):
    """Lloyd's algorithm for Voronoi cell equalization."""
    from scipy.spatial import Voronoi, QhullError
    
    p = p.copy()
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            vor = Voronoi(p)
        except QhullError:
            break
            
        new_p = p.copy()
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                new_p[i] = 0.95 * p[i]
            else:
                vertices = vor.vertices[region]
                centroid = np.mean(vertices, axis=0)
                new_p[i] = centroid
        
        margin = 1e-6
        new_p = np.clip(new_p, -0.5 + margin, 0.5 - margin)
        
        dist_to_wall = 0.5 - np.max(np.abs(new_p), axis=1)
        close_to_wall = dist_to_wall < 0.02
        if np.any(close_to_wall):
            center = np.zeros(3)
            new_p[close_to_wall] += 0.03 * (center - new_p[close_to_wall])
            new_p = np.clip(new_p, -0.5 + margin, 0.5 - margin)

        p = new_p
    
    return p

def _active_set_polish(p, target_r, max_iters=200, time_limit=None, start_time=None):
    """
    Active-set polishing: identify the constraints that are tight (active)
    and solve a local optimization on just those variables.
    """
    p = p.copy()
    current_r = _compute_radius(p)
    
    if current_r < target_r - 1e-10:
        return p, current_r
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Find active constraints
        dists = cdist(p, p, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        b = 0.5 - np.max(np.abs(p), axis=1)
        
        # Active pairs: distance < 2*(target_r + 1e-8)
        active_pairs = np.argwhere(dists < 2 * (target_r + 1e-8))
        active_pairs = [(i, j) for i, j in active_pairs if i < j]
        
        # Active boundaries: b < target_r + 1e-8
        active_bounds = np.where(b < target_r + 1e-8)[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Try to improve by moving points that are in active constraints
        improved = False
        for i, j in active_pairs[:20]:  # Limit to 20 pairs for speed
            # Try small moves to increase the minimum distance
            mid = (p[i] + p[j]) / 2
            direction = p[i] - p[j]
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-10:
                continue
            direction = direction / norm_dir
            
            # Move i and j apart slightly
            for delta in [0.0005, 0.0002, 0.0001]:
                p_trial = p.copy()
                p_trial[i] += delta * direction
                p_trial[j] -= delta * direction
                p_trial = np.clip(p_trial, -0.5, 0.5)
                
                r_trial = _compute_radius(p_trial)
                if r_trial > current_r + 1e-12:
                    p = p_trial
                    current_r = r_trial
                    improved = True
                    break
        
        if not improved:
            # Try boundary adjustments
            for i in active_bounds[:10]:
                for axis in range(3):
                    # Push away from wall
                    wall_dir = -np.sign(p[i, axis])
                    for delta in [0.0005, 0.0002, 0.0001]:
                        p_trial = p.copy()
                        p_trial[i, axis] += delta * wall_dir
                        p_trial = np.clip(p_trial, -0.5, 0.5)
                        
                        r_trial = _compute_radius(p_trial)
                        if r_trial > current_r + 1e-12:
                            p = p_trial
                            current_r = r_trial
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return p, current_r

def _local_search(p0, time_limit, start_time):
    """
    Combined local search with multiple methods.
    """
    p = p0.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    betas = [20.0, 50.0, 100.0, 300.0, 1000.0]
    
    while time.time() - start_time < time_limit:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3.0:
            break
        
        # 1. Vectorized repulsion anneal (fast, good for global improvement)
        if np.random.random() < 0.3:
            p_rep = _vectorized_repulsion_anneal(p, n_steps=50, lr_start=0.015, lr_end=0.002,
                                                  time_limit=remaining, start_time=start_time)
            r_rep = _compute_radius(p_rep)
            
            if r_rep > best_r:
                best_r = r_rep
                best_p = p_rep.copy()
                p = p_rep
        
        # 2. Soft-min polish with analytic gradients
        improved_in_polish = False
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
            
            def obj_and_grad(x, _beta=beta):
                val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=_beta)
                return val, grad
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                result = minimize(obj_and_grad, p.flatten(), method='L-BFGS-B',
                                 jac=True, bounds=bounds,
                                 options={'maxiter': 50, 'gtol': 1e-7, 'maxls': 15})
                p_smooth = result.x.reshape(N, 3)
                r_smooth = _compute_radius(p_smooth)
                
                if r_smooth > best_r + 1e-10:
                    best_r = r_smooth
                    best_p = p_smooth.copy()
                    p = p_smooth
                    improved_in_polish = True
            except Exception:
                pass
        
        # 3. Lloyd step (less frequent)
        if np.random.random() < 0.2:
            p_lloyd = _lloyd_step(p, n_steps=1, time_limit=remaining, start_time=start_time)
            r_lloyd = _compute_radius(p_lloyd)
            
            if r_lloyd > best_r:
                best_r = r_lloyd
                best_p = p_lloyd.copy()
                p = p_lloyd
                improved_in_polish = True
        
        # 4. If no improvement, try Metropolis MC to escape local minima
        if not improved_in_polish:
            p_mc, r_mc = _metropolis_mc(best_p, n_steps=15000,
                                        time_limit=remaining, start_time=start_time)
            if r_mc > best_r + 1e-10:
                best_r = r_mc
                best_p = p_mc.copy()
                p = p_mc
            else:
                if time.time() - start_time > time_limit * 0.8:
                    break
                p = best_p + np.random.normal(0, 0.002, size=(N, 3))
                p = np.clip(p, -0.5, 0.5)
    
    return best_p, best_r

def _seed_fcc():
    """Generate FCC lattice points."""
    points = []
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    return points[idx[:N]].copy()

def _seed_grid():
    """Generate a 3D grid of points."""
    coords = np.linspace(-0.4, 0.4, 4)
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
    grid -= np.mean(grid, axis=0)
    
    dists = np.linalg.norm(grid, axis=0)
    idx = np.argsort(dists)
    
    return grid[idx[:N]].copy()

def _seed_random(seed=42):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(-0.4, 0.4, size=(N, 3))
    return p

def _final_polish(p, time_limit, start_time):
    """
    Final high-precision polish with multi-stage soft-min and active-set.
    """
    p = p.copy()
    best_r = _compute_radius(p)
    best_p = p.copy()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Stage 1: moderate beta
    if time.time() - start_time < time_limit - 10.0:
        try:
            def obj1(x):
                val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=200.0)
                return val, grad
            result = minimize(obj1, p.flatten(), method='L-BFGS-B',
                             jac=True, bounds=bounds,
                             options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 20})
            p = result.x.reshape(N, 3)
            r = _compute_radius(p)
            if r > best_r:
                best_r = r
                best_p = p.copy()
        except Exception:
            pass
    
    # Stage 2: high beta
    if time.time() - start_time < time_limit - 5.0:
        try:
            def obj2(x):
                val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=1000.0)
                return val, grad
            result = minimize(obj2, p.flatten(), method='L-BFGS-B',
                             jac=True, bounds=bounds,
                             options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 30})
            p = result.x.reshape(N, 3)
            r = _compute_radius(p)
            if r > best_r:
                best_r = r
                best_p = p.copy()
        except Exception:
            pass
    
    # Stage 3: very high beta
    if time.time() - start_time < time_limit - 3.0:
        try:
            def obj3(x):
                val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=5000.0)
                return val, grad
            result = minimize(obj3, p.flatten(), method='L-BFGS-B',
                             jac=True, bounds=bounds,
                             options={'maxiter': 100, 'gtol': 1e-12, 'maxls': 20})
            p = result.x.reshape(N, 3)
            r = _compute_radius(p)
            if r > best_r:
                best_r = r
                best_p = p.copy()
        except Exception:
            pass
    
    # Active-set polish
    if time.time() - start_time < time_limit - 2.0:
        remaining = time_limit - 2.0 - (time.time() - start_time)
        p_as, r_as = _active_set_polish(best_p, best_r, max_iters=100,
                                         time_limit=remaining, start_time=start_time)
        if r_as > best_r:
            best_r = r_as
            best_p = p_as
    
    return best_p, best_r

def construct():
    start_time = time.time()
    
    seeds = [
        ('fcc', _seed_fcc),
        ('grid', _seed_grid),
        ('random', _seed_random),
    ]
    
    best_p = None
    best_r = -1
    
    # Allocate time: give more to FCC (usually best)
    time_allocations = [0.55, 0.25, 0.20]
    available_time = TIME_LIMIT - 15.0  # Reserve 15s for final polish
    
    for i, (seed_name, seed_func) in enumerate(seeds):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15.0:
            break
        
        p0 = seed_func()
        
        max_coord = np.max(np.abs(p0))
        if max_coord > 0:
            scale = 0.42 / max_coord
            p0 = p0 * scale
        
        seed_time_limit = time_allocations[i] * available_time
        seed_time_limit = min(seed_time_limit, TIME_LIMIT - 15.0 - elapsed)
        if seed_time_limit < 5.0:
            break
        
        p_improved, r_improved = _local_search(p0, seed_time_limit, start_time)
        
        if r_improved > best_r:
            best_r = r_improved
            best_p = p_improved
    
    # Final polish
    if best_p is not None and time.time() - start_time < TIME_LIMIT - 5.0:
        remaining = TIME_LIMIT - 5.0 - (time.time() - start_time)
        best_p, best_r = _final_polish(best_p, remaining, start_time)
    
    # Final validation
    if best_p is not None:
        final_r = _compute_radius(best_p)
        if final_r > best_r:
            best_r = final_r
    
    # Safety: ensure we return before time limit
    if time.time() - start_time > TIME_LIMIT - 1.0:
        pass  # Already close to limit, just return
    
    return best_p
