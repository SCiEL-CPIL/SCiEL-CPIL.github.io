import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree

N = 50
HALF = 0.5

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return max(0.0, 140.0 - _elapsed())

_start_time = time.time()

def compute_radius(centers):
    """
    Compute the maximum radius for the given centers.
    Returns (radius, valid)
    """
    if centers.shape != (N, 2):
        return 0.0, False
    if not np.all(np.isfinite(centers)):
        return 0.0, False
    if np.any(centers < -HALF + 1e-12) or np.any(centers > HALF - 1e-12):
        return 0.0, False
        
    d = pdist(centers)
    min_pair = d.min() if d.size > 0 else 0.0
    dist_walls = np.min(np.stack([
        centers[:, 0] + HALF,
        HALF - centers[:, 0],
        centers[:, 1] + HALF,
        HALF - centers[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r, True

def penalty_function(params, R):
    """
    Penalty function for radius R.
    params: flattened array of length 2*N
    """
    centers = params.reshape(N, 2)
    
    d_x = 0.5 - np.abs(centers[:, 0])
    d_y = 0.5 - np.abs(centers[:, 1])
    
    wall_viol_x = np.maximum(0, R - d_x)
    wall_viol_y = np.maximum(0, R - d_y)
    
    wall_penalty = np.sum(wall_viol_x**2 + wall_viol_y**2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i, j = np.triu_indices(N, k=1)
    pair_dists = dists[i, j]
    
    pair_viol = np.maximum(0, 2*R - pair_dists)
    pair_penalty = np.sum(pair_viol**2)
    
    return wall_penalty + pair_penalty

def _negative_smooth_score(c_flat, beta=100.0):
    """Soft-min objective for L-BFGS-B."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1).flatten()
    all_dists = np.concatenate([d, wall_dists])
    min_d = all_dists.min()
    # Soft min: min_d - log(sum(exp(-beta*(d - min_d))))/beta
    exp_d = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_d.sum()) / beta
    return -soft_min

def _polish_smooth(c, max_iter=500):
    """High-precision smooth polish with increasing beta."""
    x0 = c.ravel()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    best_c = c.copy()
    best_r, _ = compute_radius(best_c)
    
    # Increase beta to sharpen the soft-min
    for beta in [1000.0, 2000.0, 5000.0, 10000.0, 20000.0]:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                lambda cf, b=beta: _negative_smooth_score(cf, b),
                x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16, 'iprint': 0}
            )
            x0 = res.x
            c_new = np.clip(x0.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            r_new, valid = compute_radius(c_new)
            if valid and r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            pass
            
    return best_c

def _active_set_slsqp(c, r_target, max_iter=100):
    """
    SLSQP on the active constraint subset: maximize t subject to
    active pairwise and wall constraints >= t.
    """
    # Identify active constraints
    D = cdist(c, c)
    np.fill_diagonal(D, np.inf)
    half_D = D / 2.0
    iu = np.triu_indices(N, k=1)
    pair_dists = half_D[iu]
    
    tol = 1e-6
    active_pairs = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(iu[0]))
                    if pair_dists[k] <= r_target + tol]
    
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    active_walls = []
    for i in range(N):
        for w in range(4):
            if wall_dists[i, w] <= r_target + tol:
                active_walls.append((i, w))
    
    if len(active_pairs) == 0 and len(active_walls) == 0:
        return c
    
    def cons(x):
        cc = x[:2*N].reshape(N, 2)
        t = x[2*N]
        vals = []
        for (i, j) in active_pairs:
            vals.append(np.linalg.norm(cc[i] - cc[j]) / 2.0 - t)
        for (i, w) in active_walls:
            if w == 0:
                vals.append(cc[i, 0] + HALF - t)
            elif w == 1:
                vals.append(HALF - cc[i, 0] - t)
            elif w == 2:
                vals.append(cc[i, 1] + HALF - t)
            else:
                vals.append(HALF - cc[i, 1] - t)
        return np.array(vals)
    
    def obj(x):
        return -x[2*N]
    
    x0 = np.concatenate([c.ravel(), [r_target]])
    bounds = [(-HALF + 1e-10, HALF - 1e-10)] * (2*N) + [(0.0, HALF)]
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': cons}],
            options={'maxiter': max_iter, 'ftol': 1e-14, 'iprint': 0}
        )
        if res.success:
            cc = np.clip(res.x[:2*N].reshape(N, 2), -HALF + 1e-10, HALF - 1e-10)
            r_new, valid = compute_radius(cc)
            r_old, _ = compute_radius(c)
            if valid and r_new >= r_old:
                return cc
    except Exception:
        pass
    return c

def _grid_init():
    """Grid initial layout."""
    # 7x7 grid covers 49 points, add 1 more? Or 7x8?
    # 7x7 = 49. 50th point needs to be added.
    # Let's use a 7x7 grid and add a point in the center of a cell?
    # Or just use linspace for 7x7 and duplicate one? No, that's bad.
    # Let's use 7 columns and 8 rows? 56 points. Pick 50.
    # Or 6x9?
    # Best known is close to 7x7 with some shifts.
    
    # Simple: 7x7 grid, take first 49, add 1 random?
    # Better: 7x8 grid, pick 50 closest to center?
    
    # Let's stick to the 7x7 idea but perturb it.
    # Actually, the current program used m=ceil(sqrt(50))=8. 8x8=64.
    # Let's use 7x7 and add one point in a gap.
    
    # For robustness, let's generate a 7x7 grid and one extra point.
    # Or just use the 8x8 grid and pick 50 best?
    
    # Let's use 7x7 grid centered.
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We have 49 points. Add a 50th.
    # Where to add it? In the middle of one of the cells?
    # Or just return 49? No, N=50.
    
    # Let's use 7x8 grid (56 points) and pick 50 closest to center.
    m_r, m_c = 7, 8
    x = (np.arange(m_c) + 0.5) / m_c - 0.5
    y = (np.arange(m_r) + 0.5) / m_r - 0.5
    full_grid = np.array([(xi, yi) for yi in y for xi in x])
    
    dists = np.sum(full_grid**2, axis=1)
    sorted_idx = np.argsort(dists)
    return full_grid[sorted_idx[:N]]

def construct():
    global _start_time
    _start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    # Initial grid
    initial_centers = _grid_init()
    r0, _ = compute_radius(initial_centers)
    if r0 > best_radius:
        best_radius = r0
        best_centers = initial_centers.copy()
    
    # Jitter for multiple restarts
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    # Optimization settings
    options = {'maxiter': 300, 'ftol': 1e-10}
    
    # We will try to increase R from a low value
    # Start with R slightly above grid radius
    R_low = r0 * 1.001
    R_high = 0.072
    
    current_centers = initial_centers.copy()
    
    # To make it faster, we can do a coarse search first
    # Or just iterate R up
    
    # Let's do a simple hill-climbing on R
    R = R_low
    
    for restart in range(num_restarts):
        if _remaining() < 15.0:
            break
            
        if restart > 0:
            # Perturb the best found so far
            if best_centers is not None:
                current_centers = best_centers + rng.normal(0, 0.005, (N, 2))
                current_centers = np.clip(current_centers, -0.45, 0.45)
            else:
                current_centers = initial_centers + rng.normal(0, 0.01, (N, 2))
                current_centers = np.clip(current_centers, -0.45, 0.45)
            
            # Reset R to a value slightly below the best found
            if best_radius > 0:
                R = best_radius * 0.99
            else:
                R = R_low
                
        # Optimization loop for this restart
        while R < R_high:
            if _remaining() < 10.0:
                break
                
            success = False
            best_pen = np.inf
            best_x = current_centers.flatten()
            
            # Try a few random perturbations of the current best as start points
            starts = [current_centers.flatten()]
            
            # Add a few random perturbations if we have time
            if _remaining() > 30.0:
                for _ in range(2):
                    perturbed = current_centers + rng.normal(0, 0.01, (N, 2))
                    perturbed = np.clip(perturbed, -0.45, 0.45)
                    starts.append(perturbed.flatten())
                    
            for x0 in starts:
                if _remaining() < 10.0:
                    break
                    
                try:
                    res = minimize(
                        penalty_function, 
                        x0, 
                        args=(R,), 
                        method='L-BFGS-B', 
                        bounds=[(-0.5, 0.5)] * (2*N),
                        options=options
                    )
                    
                    if res.fun < best_pen:
                        best_pen = res.fun
                        best_x = res.x
                        
                    # If penalty is very low, we might have found a feasible solution
                    if best_pen < 1e-8:
                        success = True
                        break
                except Exception:
                    continue
                    
            if success:
                # Update best centers
                current_centers = best_x.reshape(N, 2)
                # Calculate actual radius to update best
                r_actual, valid = compute_radius(current_centers)
                if valid and r_actual > best_radius:
                    best_radius = r_actual
                    best_centers = current_centers.copy()
                    
                # Increase R
                R += 0.0005
            else:
                # If not feasible, we might be stuck or R is too high
                break

    # Final Polish Phase
    if best_centers is not None and _remaining() > 10.0:
        # 1. Smooth Polish
        polished = _polish_smooth(best_centers, max_iter=500)
        r_polished, valid = compute_radius(polished)
        if valid and r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
        
        # 2. Active Set SLSQP Polish
        if _remaining() > 5.0:
            refined = _active_set_slsqp(best_centers, best_radius, max_iter=100)
            r_refined, valid = compute_radius(refined)
            if valid and r_refined > best_radius:
                best_radius = r_refined
                best_centers = refined
                
            # One more smooth polish after SLSQP?
            if _remaining() > 5.0:
                polished2 = _polish_smooth(best_centers, max_iter=200)
                r_polished2, valid = compute_radius(polished2)
                if valid and r_polished2 > best_radius:
                    best_radius = r_polished2
                    best_centers = polished2

    if best_centers is None:
        return initial_centers
        
    return best_centers

# The evaluator calls construct().
