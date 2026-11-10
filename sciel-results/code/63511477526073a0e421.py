import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 20.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute turn matrix
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # Precompute turn from pivot p
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # DP
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        
        for s in range(m):
            valid_c = np.where(turn_p[s, s + 1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s + 1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    vec = dp_prev[s, s:p_idx, p_idx]
                    if np.any(vec > 0):
                        mask = (turn_mat[s:p_idx, p_idx, p_idx + 1:] > 1e-9).astype(np.int64)
                        contrib = vec @ mask
                        dp_curr[s, p_idx, p_idx + 1:] = contrib
            dp_prev = dp_curr

        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue

                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue

                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= 1e-9:
                        continue

                    total += count

    return total

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _get_grid_positions():
    """Generate a sparse grid of high-symmetry lattice positions."""
    # Use a 6x6 grid with some jitter to avoid exact collinearity issues
    # but keep points close to grid lines for maximum collinearity
    grid_size = 6
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = 0.1 + 0.8 * i / (grid_size - 1)
            y = 0.1 + 0.8 * j / (grid_size - 1)
            positions.append((x, y))
    return np.array(positions)

def _generate_grid_initial(rng):
    """Initialize points on a sparse grid to maximize collinearity."""
    grid = _get_grid_positions()
    n_grid = len(grid)
    
    # Assign 33 points to 36 grid positions (3 remain unused)
    # Use random assignment with some structure
    assignments = []
    used = set()
    
    # First, place points on distinct grid positions
    perm = rng.permutation(n_grid)
    for idx in range(33):
        pos_idx = perm[idx]
        assignments.append(pos_idx)
    
    pts = grid[assignments].copy()
    
    # Add small perturbations to break exact collinearity
    pts += rng.normal(0, 0.002, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    # Validate and fix if needed
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 200:
        # Perturb random points more
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(33, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.01, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _hill_climb_grid(pts, rng, time_limit):
    """Hill-climb with preference for grid-snapping."""
    grid = _get_grid_positions()
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 80
    step_size = 0.03
    grid_snap_prob = 0.4
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Choose strategy: grid-snap or random perturbation
        if rng.random() < grid_snap_prob:
            # Grid-snap: move one point to a nearby grid position
            i = rng.integers(0, 33)
            # Find nearest grid positions
            dists = np.sum((grid - best_pts[i])**2, axis=1)
            nearest = np.argsort(dists)[:5]
            target_idx = nearest[rng.integers(0, 5)]
            new_pts = best_pts.copy()
            new_pts[i] = grid[target_idx] + rng.normal(0, 0.003, 2)
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        else:
            # Random perturbation
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size *= 0.95
            elif c == best_c and best_c > 1:
                if rng.random() < 0.35:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        # Multi-point kick every 15 no-improve steps
        if no_improve > 0 and no_improve % 15 == 0:
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                if rng.random() < grid_snap_prob:
                    dists = np.sum((grid - best_pts[i])**2, axis=1)
                    nearest = np.argsort(dists)[:5]
                    target_idx = nearest[rng.integers(0, 5)]
                    new_pts[i] = grid[target_idx] + rng.normal(0, 0.005, 2)
                else:
                    offset = rng.normal(0, step_size * 2, 2)
                    new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size *= 0.95
        
        if no_improve % 50 == 0:
            step_size = min(0.15, step_size * 1.3)
            grid_snap_prob = max(0.2, grid_snap_prob - 0.05)

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
        
    # Start with grid-based initialization
    best_pts = _generate_grid_initial(rng)
    best_c = _count_convex7_fast(best_pts)
    
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c
    
    print(f"Initial C: {best_c}")
    
    while time_left() > 60:
        chunk_time = min(45, time_left() - 60)
        if chunk_time < 10:
            break
            
        best_pts, best_c = _hill_climb_grid(best_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
            
        print(f"After hill climb: C={best_c}, best_ever={best_ever_c}, t={time.time() - start_time:.1f}")
        
        if best_ever_c <= 1:
            break
            
        # Restart with new grid assignment if time permits
        if time_left() > 90:
            new_pts = _generate_grid_initial(rng)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                best_pts = new_pts
                best_c = new_c
                best_ever_pts = new_pts.copy()
                best_ever_c = new_c
                print(f"Restart improved to C: {best_c}")
            else:
                best_pts = best_ever_pts.copy()
                best_c = best_ever_c

    return best_ever_pts
