"""Pack 50 equal circles in a unit square, maximize common radius.
Strategy: Hard-Disk Radius Expansion + SLSQP Epigraph Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)


def _make_grid_rect():
    """Create a 7x8 rectangular grid (56 points), take first 50."""
    nx, ny = 7, 8
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    grid = np.array([(x, y) for y in ys for x in xs])[:N]
    return grid


def _make_hexagonal_grid():
    """Create a hexagonal packing of 50 points."""
    spacing = 0.142
    dy = spacing * np.sqrt(3) / 2
    points = []
    for j in range(12):
        for i in range(12):
            x = i * spacing - 0.5
            y = j * dy - 0.5
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)[:N]
    # Center and scale to fit
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points *= (0.45 / max_ext)
    return points


def _expand_radius(centers, r_init, r_max=0.08, dr=0.0005, max_iters=2000, time_limit=30.0):
    """
    Hard-disk radius expansion: gradually increase r, resolving collisions
    by pushing overlapping circles apart.
    """
    start_time = time.time()
    centers = centers.copy()
    r = r_init
    
    # Precompute pair indices
    iu = np.triu_indices(N, 1)
    
    while r < r_max and max_iters > 0:
        if time.time() - start_time > time_limit:
            break
        
        r += dr
        max_iters -= 1
        
        # Iterate collision resolution
        for _ in range(50):
            moved = False
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                d = centers[i] - centers[j]
                dist = np.sqrt(np.dot(d, d))
                min_dist = 2.0 * r
                
                if dist < min_dist:
                    if dist < 1e-10:
                        # Coincident: nudge randomly
                        d = np.random.randn(2) * 0.001
                        dist = np.sqrt(np.dot(d, d))
                    
                    overlap = min_dist - dist
                    push = d / dist * overlap * 0.5
                    centers[i] += push
                    centers[j] -= push
                    moved = True
            
            # Clamp to square
            centers = np.clip(centers, -0.5 + r, 0.5 - r)
            
            if not moved:
                break
        
        # If we can't resolve, try smaller step
        if _radius(centers) < r:
            r -= dr
            # Try to repair
            for _ in range(20):
                for idx in range(len(iu[0])):
                    i, j = iu[0][idx], iu[1][idx]
                    d = centers[i] - centers[j]
                    dist = np.sqrt(np.dot(d, d))
                    min_dist = 2.0 * r
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        push = d / dist * overlap * 0.3
                        centers[i] += push
                        centers[j] -= push
                centers = np.clip(centers, -0.5 + r, 0.5 - r)
            if _radius(centers) < r:
                break
    
    return centers, r


def _slsqp_polish(centers, time_limit=30.0):
    """
    SLSQP optimization of the exact min-radius objective with box constraints.
    Maximize r = min(wall_r, min_pair/2) subject to box constraints.
    We minimize -r.
    """
    start_time = time.time()
    x0 = centers.ravel()
    
    iu = np.triu_indices(N, 1)
    
    def objective(x):
        c = x.reshape(N, 2)
        # Wall radius
        wall_r = np.min(0.5 - np.abs(c))
        # Pair distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        min_pair = np.min(dist[iu])
        r = min(wall_r, min_pair / 2.0)
        return -r
    
    def gradient(x):
        c = x.reshape(N, 2)
        wall_r = np.min(0.5 - np.abs(c))
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        min_pair = np.min(dist[iu])
        r = min(wall_r, min_pair / 2.0)
        
        # Gradient is complex for min; use finite differences if needed
        # For SLSQP, we can skip providing gradient and let it use finite diffs
        return None
    
    # Box constraints: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers


def _local_refine_lbfgs(centers, time_limit=10.0, beta=80.0):
    """Quick L-BFGS-B refinement with soft-min as backup."""
    start_time = time.time()
    x0 = centers.ravel()
    
    iu = np.triu_indices(N, 1)
    
    def soft_min_radius(x, beta_val):
        c = x.reshape(N, 2)
        c1 = 0.5 - c[:, 0]
        c2 = 0.5 + c[:, 0]
        c3 = 0.5 - c[:, 1]
        c4 = 0.5 + c[:, 1]
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        pair_dists = dist[iu] / 2.0
        all_c = np.concatenate([c1, c2, c3, c4, pair_dists])
        y = -beta_val * all_c
        max_y = np.max(y)
        sum_exp = np.sum(np.exp(y - max_y))
        log_sum_exp = max_y + np.log(sum_exp)
        return -log_sum_exp / beta_val
    
    def objective(x):
        return -soft_min_radius(x, beta)
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
    
    try:
        res = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers


def construct():
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Generate multiple structured initializations
    inits = []
    inits.append(_make_grid_rect())
    inits.append(_make_hexagonal_grid())
    
    # Add a few random perturbations of the grid
    np.random.seed(123)
    for _ in range(3):
        g = _make_grid_rect().copy()
        g += np.random.randn(N, 2) * 0.01
        g = np.clip(g, -0.45, 0.45)
        inits.append(g)
    
    # Phase 2: Radius expansion on each init, keep best
    # Allocate ~50% of time to expansion phase
    expand_budget = 0.5 * DEADLINE
    per_expand = expand_budget / len(inits)
    
    for i, init in enumerate(inits):
        time_left = start_time + DEADLINE - time.time()
        if time_left < 5:
            break
        alloc = min(per_expand, time_left - 5)
        if alloc < 2:
            break
        
        expanded, r_exp = _expand_radius(init, r_init=0.06, r_max=0.075, dr=0.0003,
                                          max_iters=5000, time_limit=alloc)
        r = _radius(expanded)
        if r > best_r:
            best_r = r
            best_centers = expanded
    
    if best_centers is None:
        best_centers = inits[0]
        best_r = _radius(best_centers)
    
    # Phase 3: SLSQP polish on the best candidate
    time_left = DEADLINE - 5 - (time.time() - start_time)
    if time_left > 10:
        slsqp_alloc = min(40.0, time_left * 0.6)
        polished = _slsqp_polish(best_centers, time_limit=slsqp_alloc)
        r = _radius(polished)
        if r > best_r:
            best_r = r
            best_centers = polished
    
    # Phase 4: L-BFGS-B refinement as a final polish
    time_left = DEADLINE - 3 - (time.time() - start_time)
    if time_left > 5:
        refined = _local_refine_lbfgs(best_centers, time_limit=min(15.0, time_left * 0.7), beta=100.0)
        r = _radius(refined)
        if r > best_r:
            best_r = r
            best_centers = refined
    
    # Phase 5: If time remains, try SLSQP on a second-best init
    time_left = DEADLINE - 3 - (time.time() - start_time)
    if time_left > 15:
        # Try the hexagonal init with SLSQP directly
        hex_init = _make_hexagonal_grid()
        r_hex = _radius(hex_init)
        if r_hex < best_r:
            # Quick expand then SLSQP
            t_exp = min(10.0, time_left * 0.3)
            t_slsqp = min(20.0, time_left * 0.5)
            if t_exp > 2 and t_slsqp > 5:
                exp_hex, _ = _expand_radius(hex_init, r_init=0.06, r_max=0.072,
                                             dr=0.0003, max_iters=3000, time_limit=t_exp)
                r_e = _radius(exp_hex)
                if r_e > best_r:
                    best_r = r_e
                    best_centers = exp_hex
                
                time_left = DEADLINE - 3 - (time.time() - start_time)
                if time_left > 5:
                    pol_hex = _slsqp_polish(best_centers, time_limit=min(15.0, time_left * 0.7))
                    r_p = _radius(pol_hex)
                    if r_p > best_r:
                        best_r = r_p
                        best_centers = pol_hex
    
    # Final safety: ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Verify validity
    r_final = _radius(best_centers)
    if r_final <= 0 or not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        best_centers = _make_grid_rect()
    
    return best_centers
