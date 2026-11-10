# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    Points are sorted lexicographically, for each pivot the later points are
    sorted by atan2 angle around it, and a turn p -> q -> r counts as convex
    only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    """
    n = len(points)
    if n < 7:
        return 0
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    TOL = 1e-9

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

        # Precompute turn(a,b,c) for all a<b<c
        turn_ok = np.zeros((m, m, m), dtype=bool)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_ok[a, b, b+1:] = cross_vals[b+1:] > TOL

        # Precompute turn from pivot p to q[s] to q[c]
        turn_p = np.zeros((m, m), dtype=bool)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            cross_vals = v1x * v2y - v1y * v2x
            turn_p[s, s+1:] = cross_vals[s+1:] > TOL

        # dp[s][p_idx][c_idx] = number of convex chains of current length
        # starting at index s, with previous point at p_idx, ending at c_idx
        dp_prev = np.zeros((m, m, m), dtype=np.int64)

        # Initialize: chains of length 2 (just p -> q[s])
        # For each s, mark valid c where turn(p, q[s], q[c]) > TOL
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:])[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        # Extend chains to length 3, 4, 5, 6
        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    if not np.any(dp_slice > 0):
                        continue
                    turn_slice = turn_ok[s:p_idx, p_idx, p_idx+1:]
                    contrib = (dp_slice[:, np.newaxis] * turn_slice).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev = dp_curr

        # Count complete 7-gons: need to close the chain
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue

                    # Check closing turn 1: (q[p_idx], q[c_idx], p)
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= TOL:
                        continue

                    # Check closing turn 2: (q[c_idx], p, q[s])
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= TOL:
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
    """
    Generate a set of points designed to minimize convex 7-gons.
    Strategy: Place points near a concave curve (parabola) so that most
    subsets are not in convex position. Points near a curve have the property
    that few 7-subsets form convex heptagons.
    """
    pts = []
    
    # Place points along a parabolic arc: y = a*x^2 + b*x + c
    # This creates a "concave" cluster where convex position is rare
    n_per_cluster = 11
    n_clusters = 3
    
    # Cluster 1: near bottom-left, along a curve
    for i in range(n_per_cluster):
        t = i / (n_per_cluster - 1)
        x = 0.1 + 0.3 * t
        y = 0.1 + 0.2 * t * t
        x += rng.normal(0, 0.015)
        y += rng.normal(0, 0.015)
        pts.append((x, y))
    
    # Cluster 2: near top-right, along a curve
    for i in range(n_per_cluster):
        t = i / (n_per_cluster - 1)
        x = 0.6 + 0.3 * t
        y = 0.7 + 0.2 * (1 - t) * (1 - t)
        x += rng.normal(0, 0.015)
        y += rng.normal(0, 0.015)
        pts.append((x, y))
    
    # Cluster 3: center, along a curve
    for i in range(n_per_cluster):
        t = i / (n_per_cluster - 1)
        x = 0.35 + 0.3 * t
        y = 0.45 + 0.1 * np.sin(t * np.pi)
        x += rng.normal(0, 0.015)
        y += rng.normal(0, 0.015)
        pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 200:
        pts += rng.normal(0, 0.002, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _hill_climb(pts, rng, time_limit):
    """
    Perform hill climbing to minimize the number of convex 7-gons.
    Uses adaptive step sizes and multiple-point moves to escape local minima.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 300
    step_size = 0.05
    min_step = 0.003
    max_step = 0.25
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        # Random new position within a neighborhood
        offset = rng.normal(0, step_size, 2)
        new_pts[i] = best_pts[i] + offset
        new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                # Reduce step size on improvement
                step_size = max(min_step, step_size * 0.92)
            elif c == best_c and best_c > 1:
                # Accept neutral moves to explore
                if rng.random() < 0.25:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # If stuck, try a larger jump with multiple points
        if not improved and no_improve % 3 == 0:
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 2.0, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size = max(min_step, step_size * 0.92)
                elif c == best_c and best_c > 1:
                    if rng.random() < 0.2:
                        best_pts = new_pts
                        no_improve = 0
        
        # Occasionally increase step size to escape local minima
        if no_improve % 25 == 0:
            step_size = min(max_step, step_size * 1.4)
        
        # Fine-tuning phase: if step size is very small, do precise adjustments
        if step_size < 0.015 and no_improve > 40:
            # Try very fine adjustments
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            offset = rng.normal(0, 0.002, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                elif c == best_c and best_c > 1:
                    if rng.random() < 0.15:
                        best_pts = new_pts
                        no_improve = 0
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate initial points using a structured approach
    best_pts = _generate_initial_points(rng)
    best_c = _count_convex7_fast(best_pts)
    
    print(f"Initial C: {best_c}")
    
    # Perform hill climbing with multiple restarts
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c
    
    # Try multiple different initial configurations
    for restart in range(8):
        if time_left() < 40:
            break
        
        if restart > 0:
            # Generate a new initial configuration with different random seed
            rng2 = np.random.default_rng(42 + restart * 1000)
            new_pts = _generate_initial_points(rng2)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                best_pts = new_pts
                best_c = new_c
                best_ever_pts = new_pts.copy()
                best_ever_c = new_c
                print(f"Restart {restart}: new initial C={best_c}")
            else:
                best_pts = best_ever_pts.copy()
                best_c = best_ever_c
        
        # Run hill climbing
        chunk_time = min(60, time_left() - 30)
        if chunk_time < 10:
            break
        best_pts, best_c = _hill_climb(best_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
        
        print(f"After restart {restart}: C={best_c}, best_ever={best_ever_c}, t={time.time()-start_time:.1f}")
        
        if best_ever_c <= 1:
            break
    
    # Final polish: very fine-tuning with small step sizes
    if time_left() > 20:
        print("Starting final polish...")
        polish_pts = best_ever_pts.copy()
        polish_c = best_ever_c
        polish_start = time.time()
        polish_step = 0.002
        
        for _ in range(2000):
            if time.time() - polish_start > time_left() - 10:
                break
            
            # Occasionally try multi-point moves
            if rng.random() < 0.2:
                n_move = rng.integers(2, 4)
                idx = rng.choice(33, size=n_move, replace=False)
                new_pts = polish_pts.copy()
                for i in idx:
                    offset = rng.normal(0, polish_step, 2)
                    new_pts[i] = polish_pts[i] + offset
                    new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            else:
                i = rng.integers(0, 33)
                new_pts = polish_pts.copy()
                offset = rng.normal(0, polish_step, 2)
                new_pts[i] = polish_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < polish_c:
                    polish_c = c
                    polish_pts = new_pts
                    if polish_c < best_ever_c:
                        best_ever_c = polish_c
                        best_ever_pts = polish_pts.copy()
                    polish_step = max(0.0005, polish_step * 0.95)
                elif c == polish_c and polish_c > 1:
                    if rng.random() < 0.1:
                        polish_pts = new_pts
        
        print(f"After polish: C={polish_c}, best_ever={best_ever_c}")
    
    return best_ever_pts
# EVOLVE-BLOCK-END
