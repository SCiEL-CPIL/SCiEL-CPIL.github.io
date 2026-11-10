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

        # Sum up for l=6
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

def _generate_initial_points(rng):
    """Generate multiple diverse initial configurations and return the best."""
    candidates = []
    
    # Strategy 1: Concave parabola
    x_vals = np.linspace(0.05, 0.95, 33)
    y_vals = 0.5 * (1 - (2 * x_vals - 1)**2)
    x_vals += rng.normal(0, 0.003, 33)
    y_vals += rng.normal(0, 0.003, 33)
    pts1 = np.column_stack((x_vals, y_vals))
    pts1 = np.clip(pts1, 0.01, 0.99)
    candidates.append(pts1)
    
    # Strategy 2: Near-collinear with slight curvature
    x_vals = np.linspace(0.05, 0.95, 33)
    y_vals = np.full(33, 0.5) + 0.1 * np.sin(np.linspace(0, np.pi, 33))
    x_vals += rng.normal(0, 0.002, 33)
    y_vals += rng.normal(0, 0.002, 33)
    pts2 = np.column_stack((x_vals, y_vals))
    pts2 = np.clip(pts2, 0.01, 0.99)
    candidates.append(pts2)
    
    # Strategy 3: Two clusters on a line
    x1 = np.linspace(0.05, 0.45, 17)
    x2 = np.linspace(0.55, 0.95, 16)
    x_vals = np.concatenate([x1, x2])
    y_vals = np.full(33, 0.5) + rng.normal(0, 0.005, 33)
    pts3 = np.column_stack((x_vals, y_vals))
    pts3 = np.clip(pts3, 0.01, 0.99)
    candidates.append(pts3)
    
    # Strategy 4: Cubic curve
    t = np.linspace(0, 1, 33)
    x_vals = 0.05 + 0.9 * t
    y_vals = 0.5 + 0.3 * (t - 0.5)**2
    x_vals += rng.normal(0, 0.003, 33)
    y_vals += rng.normal(0, 0.003, 33)
    pts4 = np.column_stack((x_vals, y_vals))
    pts4 = np.clip(pts4, 0.01, 0.99)
    candidates.append(pts4)
    
    # Validate and pick best
    best_pts = None
    best_c = float('inf')
    for pts in candidates:
        attempts = 0
        while not _is_valid_fast(pts) and attempts < 50:
            pts = pts + rng.normal(0, 0.002, (33, 2))
            pts = np.clip(pts, 0.01, 0.99)
            attempts += 1
        if _is_valid_fast(pts):
            c = _count_convex7_fast(pts)
            if c < best_c:
                best_c = c
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random
        best_pts = rng.uniform(0.1, 0.9, (33, 2))
        best_c = _count_convex7_fast(best_pts)
    
    return best_pts

def _simulated_annealing(pts, rng, time_limit):
    """Simulated annealing to minimize convex 7-gon count."""
    best_pts = pts.copy()
    current_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    current_c = best_c
    
    start_time = time.time()
    T = 0.5  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        n_move = rng.integers(1, 5)  # Move 1-4 points
        idx = rng.choice(33, size=n_move, replace=False)
        
        new_pts = current_pts.copy()
        step = T * 0.1 + 0.01
        for i in idx:
            offset = rng.normal(0, step, 2)
            new_pts[i] = current_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if not _is_valid_fast(new_pts):
            T *= cooling_rate
            continue
        
        new_c = _count_convex7_fast(new_pts)
        delta_c = new_c - current_c
        
        # Acceptance criterion
        if delta_c < 0:
            # Always accept improvements
            current_pts = new_pts
            current_c = new_c
            if current_c < best_c:
                best_c = current_c
                best_pts = current_pts.copy()
        else:
            # Accept worse states with probability exp(-delta_c / T)
            if T > T_min and rng.random() < np.exp(-delta_c / T):
                current_pts = new_pts
                current_c = new_c
        
        T *= cooling_rate
        
        # Occasionally reset to best to avoid getting stuck
        if rng.random() < 0.01:
            current_pts = best_pts.copy()
            current_c = best_c
    
    return best_pts, best_c

def _hill_climb(pts, rng, time_limit):
    """Fast hill climbing as a fallback."""
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 30
    step_size = 0.03
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
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
                step_size *= 0.97
            elif c == best_c and best_c > 1:
                if rng.random() < 0.2:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        if no_improve % 15 == 0:
            n_move = rng.integers(2, 4)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 1.5, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size *= 0.97

        if no_improve % 30 == 0:
            step_size = min(0.1, step_size * 1.3)

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
        
    best_pts = _generate_initial_points(rng)
    best_c = _count_convex7_fast(best_pts)
    
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c
    
    print(f"Initial C: {best_c}")
    
    # Phase 1: Simulated annealing (main optimization)
    while time_left() > 120:
        chunk_time = min(60, time_left() - 120)
        if chunk_time < 15:
            break
            
        best_pts, best_c = _simulated_annealing(best_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
            
        print(f"After SA: C={best_c}, best_ever={best_ever_c}, t={time.time() - start_time:.1f}")
        
        if best_ever_c <= 1:
            break
    
    # Phase 2: Fine-tuning with hill climbing
    if time_left() > 60:
        best_pts, best_c = _hill_climb(best_pts, rng, min(60, time_left() - 30))
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
            print(f"After hill climb: C={best_c}, best_ever={best_ever_c}")
    
    # Phase 3: Final restarts if time permits
    if time_left() > 90:
        for _ in range(3):
            if time_left() < 45:
                break
            new_pts = _generate_initial_points(rng)
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
            
            if time_left() > 45:
                best_pts, best_c = _simulated_annealing(best_pts, rng, min(30, time_left() - 45))
                if best_c < best_ever_c:
                    best_ever_c = best_c
                    best_ever_pts = best_pts.copy()
    
    return best_ever_pts
