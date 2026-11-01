import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(points):
    """
    Compute the maximum radius such that circles centered at points
    are inside [-0.5, 0.5]^2 and non-overlapping.
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    abs_pts = np.abs(points)
    dist_x = 0.5 - abs_pts[:, 0]
    dist_y = 0.5 - abs_pts[:, 1]
    wall_dists = np.minimum(dist_x, dist_y)
    min_wall = np.min(wall_dists)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_pair / 2.0, min_wall)
    return r if r > 0 else 0.0

def _energy(points, r):
    """
    Calculate energy for a given configuration and target radius r.
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    abs_pts = np.abs(points)
    dist_x = 0.5 - abs_pts[:, 0]
    dist_y = 0.5 - abs_pts[:, 1]
    wall_dists = np.minimum(dist_x, dist_y)
    wall_viol = np.maximum(0, r - wall_dists)
    wall_energy = np.sum(wall_viol ** 2)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0, 2.0 * r - dists)
    pair_energy = np.sum(pair_viol ** 2)
    
    return wall_energy + pair_energy

def _sa_optimize(init_points, r_target, time_limit):
    points = init_points.copy()
    T = 0.01
    T_end = 1e-6
    cooling = 0.9995
    step_size = 0.02
    
    current_energy = _energy(points, r_target)
    best_points = points.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    while T > T_end and (time.time() - start_time) < time_limit:
        idx = np.random.randint(len(points))
        orig = points[idx].copy()
        
        delta = np.random.normal(0, step_size, size=2)
        points[idx] += delta
        points[idx] = np.clip(points[idx], -0.5, 0.5)
        
        new_energy = _energy(points, r_target)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
        else:
            points[idx] = orig
            
        T *= cooling
        
    return best_points, best_energy

def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _neg_radius(params):
    points = params.reshape(-1, 2)
    if np.any(np.abs(points) > 0.5):
        return 1e10
    r = _compute_radius(points)
    return -r

def construct():
    deadline = time.time() + 145.0
    
    best_points = _grid_init()
    best_r = _compute_radius(best_points)
    
    r_current = 0.065
    r_step = 0.001
    
    current_init = _grid_init()
    current_init += np.random.normal(0, 0.01, size=(N, 2))
    
    while time.time() < deadline:
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        alloc_time = min(10.0, time_left * 0.5)
        
        pts, energy = _sa_optimize(current_init, r_current, alloc_time)
        
        if energy < 1e-6:
            r_new = _compute_radius(pts)
            if r_new > best_r:
                best_r = r_new
                best_points = pts.copy()
            
            r_current += r_step
            current_init = pts.copy()
        else:
            if time.time() < deadline - 5:
                rand_init = np.random.uniform(-0.45, 0.45, size=(N, 2))
                pts, energy = _sa_optimize(rand_init, r_current, min(5.0, max(1.0, deadline - time.time() - 1)))
                if energy < 1e-6:
                    r_new = _compute_radius(pts)
                    if r_new > best_r:
                        best_r = r_new
                        best_points = pts.copy()
                    r_current += r_step
                    current_init = pts.copy()
                else:
                    r_current -= r_step * 2
                    if r_current < 0.05:
                        break
                    current_init = _grid_init()
                    current_init += np.random.normal(0, 0.01, size=(N, 2))

    # Final Polish with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            result = minimize(
                _neg_radius,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < -best_r:
                new_points = result.x.reshape(-1, 2)
                new_points = np.clip(new_points, -0.4999999999, 0.4999999999)
                r_new = _compute_radius(new_points)
                if r_new > best_r:
                    best_points = new_points
                    best_r = r_new
        except Exception:
            pass

    best_points = np.clip(best_points, -0.4999999999, 0.4999999999)
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        return _grid_init()
        
    return best_points

# EVOLVE-BLOCK-START
# (The above construct function is the entry point)
# EVOLVE-BLOCK-END
