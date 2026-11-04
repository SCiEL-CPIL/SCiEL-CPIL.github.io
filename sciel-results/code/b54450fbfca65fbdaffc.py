"""Circle packing in unit square – structural rewrite.

Uses a two-phase approach:
  Phase 1: Multi-start local search with "best-fit insertion" initialization
           followed by iterative pairwise radius/position adjustment.
  Phase 2: A growth polish that tries to uniformly scale radii up while
           re-relaxing positions to maintain feasibility.

Key structural ideas:
  - Initialization via sequential best-fit placement (greedy).
  - Optimization via direct geometric projection (not physics simulation).
  - Growth polish: binary-search the maximum uniform scale factor.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 26


def _elapsed(deadline):
    return time.time() - deadline + 150.0


def _remaining(deadline):
    return deadline - time.time()


def _is_valid(xy, r, tol=1e-10):
    """Check if all circles are inside [0,1]^2 and non-overlapping."""
    n = len(r)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _total_radius(r):
    return float(np.sum(r))


def _project_feasible(xy, r):
    """Project radii to be at most the clearance to walls."""
    for i in range(len(r)):
        max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        if r[i] > max_r:
            r[i] = max_r
    return r


def _max_radius_at(x, y, other_xy, other_r):
    """Maximum radius for a circle at (x,y) given others."""
    wall_clear = min(x, 1.0 - x, y, 1.0 - y)
    if wall_clear <= 0:
        return 0.0
    if len(other_r) == 0:
        return wall_clear
    dx = x - other_xy[:, 0]
    dy = y - other_xy[:, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    circle_clear = np.min(dists - other_r)
    return max(0.0, min(wall_clear, circle_clear))


def _greedy_init(n=N, seed=0):
    """
    Sequential best-fit placement:
    For each circle, find a grid candidate point that maximizes the
    resulting radius given already-placed circles.
    """
    rng = np.random.RandomState(seed)
    # Use a fine grid for candidate positions
    grid_n = 40
    xs = np.linspace(0.02, 0.98, grid_n)
    ys = np.linspace(0.02, 0.98, grid_n)
    
    placed_xy = []
    placed_r = []
    
    for k in range(n):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        # Sample a random subset of grid points to keep it fast
        n_samples = min(800, grid_n * grid_n)
        if grid_n * grid_n > n_samples:
            idx = rng.randint(0, grid_n * grid_n, size=n_samples)
            gx = (idx // grid_n).astype(float)
            gy = (idx % grid_n).astype(float)
            cx = xs[gx]
            cy = ys[gy]
        else:
            cx = np.repeat(xs, grid_n)
            cy = np.tile(ys, grid_n)
        
        if len(placed_r) == 0:
            radii = np.array([_max_radius_at(x, y, np.array([]), np.array([])) for x, y in zip(cx, cy)])
        else:
            oxy = np.array(placed_xy)
            orr = np.array(placed_r)
            radii = np.array([_max_radius_at(x, y, oxy, orr) for x, y in zip(cx, cy)])
        
        best_idx = np.argmax(radii)
        best_x, best_y, best_r = cx[best_idx], cy[best_idx], radii[best_idx]
        
        if best_r < 0.005:
            # Fallback: random small circle
            best_x = rng.uniform(0.1, 0.9)
            best_y = rng.uniform(0.1, 0.9)
            best_r = 0.01
        
        placed_xy.append((best_x, best_y))
        placed_r.append(best_r)
    
    return np.array(placed_xy), np.array(placed_r)


def _relax_positions(xy, r, n_iter=200, deadline=None):
    """
    Iteratively adjust positions to resolve overlaps while keeping radii fixed.
    Uses a simple force-directed relaxation with hard projection.
    """
    n = len(r)
    x = xy[:, 0].copy()
    y = xy[:, 1].copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-12:
                    if dist < 1e-10:
                        dx, dy = 1e-6, 1e-6
                        dist = np.sqrt(dx * dx + dy * dy)
                    overlap = min_dist - dist
                    # Push apart by half the overlap each
                    push = overlap * 0.5
                    nx, ny = dx / dist, dy / dist
                    x[i] += push * nx
                    y[i] += push * ny
                    x[j] -= push * nx
                    y[j] -= push * ny
                    moved = True
        
        # Project to walls
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])
        
        if not moved:
            break
    
    return np.column_stack([x, y])


def _grow_and_relax(xy, r, scale, n_iter=100, deadline=None):
    """
    Scale radii by `scale`, then relax positions to maintain feasibility.
    Returns (new_xy, new_r, is_valid).
    """
    new_r = r * scale
    new_xy = _relax_positions(xy, new_r, n_iter=n_iter, deadline=deadline)
    new_r = _project_feasible(new_xy, new_r)
    valid = _is_valid(new_xy, new_r)
    return new_xy, new_r, valid


def _local_optimize(xy, r, deadline, n_passes=30):
    """
    Alternating optimization:
      - Try to increase each radius individually (best-fit with others fixed).
      - Relax positions after each radius increase.
    """
    n = len(r)
    best_total = _total_radius(r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    for p in range(n_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try to grow circle i
            old_r = r[i]
            # Max radius at current position
            others_xy = np.delete(xy, i, axis=0)
            others_r = np.delete(r, i)
            max_r_here = _max_radius_at(xy[i, 0], xy[i, 1], others_xy, others_r)
            
            if max_r_here > r[i] + 1e-6:
                # Grow to max
                r[i] = max_r_here
                xy = _relax_positions(xy, r, n_iter=30, deadline=deadline)
                r = _project_feasible(xy, r)
                if _is_valid(xy, r):
                    total = _total_radius(r)
                    if total > best_total:
                        best_total = total
                        best_xy = xy.copy()
                        best_r = r.copy()
                    improved = True
                else:
                    # Revert
                    r[i] = old_r
                    xy = _relax_positions(xy, r, n_iter=20, deadline=deadline)
                    r = _project_feasible(xy, r)
        
        if not improved:
            # Try position-only relaxation to see if we can free up space
            xy = _relax_positions(xy, r, n_iter=50, deadline=deadline)
            r = _project_feasible(xy, r)
            if _is_valid(xy, r):
                total = _total_radius(r)
                if total > best_total:
                    best_total = total
                    best_xy = xy.copy()
                    best_r = r.copy()
            else:
                xy = best_xy
                r = best_r.copy()
    
    return best_xy, best_r, best_total


def _growth_polish(xy, r, deadline, n_trials=20):
    """
    Binary-search style: try to uniformly scale up radii and re-relax.
    """
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = _total_radius(r)
    
    current_scale = 1.0
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        # Try a small uniform increase
        candidate_scale = current_scale * 1.01
        new_xy, new_r, valid = _grow_and_relax(best_xy, best_r, candidate_scale,
                                                n_iter=80, deadline=deadline)
        if valid:
            new_r = _project_feasible(new_xy, new_r)
            total = _total_radius(new_r)
            if total > best_total:
                best_total = total
                best_xy = new_xy
                best_r = new_r
                current_scale = candidate_scale
            else:
                # Didn't improve, stop
                break
        else:
            # Too aggressive, reduce step
            candidate_scale = current_scale * 1.005
            new_xy, new_r, valid = _grow_and_relax(best_xy, best_r, candidate_scale,
                                                    n_iter=50, deadline=deadline)
            if valid:
                new_r = _project_feasible(new_xy, new_r)
                total = _total_radius(new_r)
                if total > best_total:
                    best_total = total
                    best_xy = new_xy
                    best_r = new_r
                    current_scale = candidate_scale
            else:
                break
    
    return best_xy, best_r, best_total


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_total = 0.0
    best_result = None

    # Multi-start with different greedy seeds
    seeds = [0, 1, 2, 3, 7, 13, 42, 100, 200, 500]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        xy, r = _greedy_init(N, seed=seed)
        
        # Quick relaxation to make it feasible
        xy = _relax_positions(xy, r, n_iter=50, deadline=deadline)
        r = _project_feasible(xy, r)
        
        if not _is_valid(xy, r):
            # Shrink radii to make valid
            shrink = 0.9
            for _ in range(10):
                r_try = r * shrink
                if _is_valid(xy, r_try):
                    r = r_try
                    break
                shrink *= 0.9
            if not _is_valid(xy, r):
                continue
        
        # Local optimization
        xy, r, total = _local_optimize(xy, r, deadline, n_passes=25)
        
        if total > best_total:
            best_total = total
            best_result = np.column_stack([xy, r])
        
        # Growth polish on the best found so far
        if best_result is not None:
            bx, by, br = best_result[:, 0], best_result[:, 1], best_result[:, 2]
            bx, by, br = _growth_polish(np.column_stack([bx, by]), br, deadline, n_trials=15)
            new_total = _total_radius(br)
            if new_total > best_total:
                best_total = new_total
                best_result = np.column_stack([bx, by, br])
    
    # Fallback
    if best_result is None:
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and cleanup
    result = best_result.copy()
    n = len(result)
    for i in range(n):
        x, y, ri = result[i, 0], result[i, 1], result[i, 2]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    if not _is_valid(result[:, :2], result[:, 2]):
        # Shrink all radii slightly
        for _ in range(5):
            result[:, 2] *= 0.95
            if _is_valid(result[:, :2], result[:, 2]):
                break
        else:
            side = 5
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
