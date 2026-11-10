import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 20.0

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
        # turn_mat[a, b, c] > 0 iff turn q[a] -> q[b] -> q[c] is convex (left turn)
        # We only need b > a and c > b.
        # Vectorized computation for all a, b, c is expensive (m^3).
        # m is max 32. 32^3 = 32768. This is fine.
        # q[:, 0] shape (m,). 
        # We can compute cross products for all pairs.
        
        # Let's optimize the turn matrix calculation.
        # cross(a, b, c) = (q[b]-q[a]) x (q[c]-q[b])
        # = (q[b][0]-q[a][0])*(q[c][1]-q[b][1]) - (q[b][1]-q[a][1])*(q[c][0]-q[b][0])
        
        # For a fixed b, and all a < b, c > b:
        # We can loop b.
        
        # dp_prev[s, p, c]
        # s < p < c
        # dp_prev is (m, m, m) but only upper triangle part is used.
        
        # We will use a sparse-like approach or just dense arrays since m is small.
        
        # Initialize dp for length 2 chains (pivot -> s -> c)
        # Actually, the code initializes dp for chains of length 2 (2 points after pivot? No.)
        # The problem is counting 7-gons. The pivot is one point.
        # The chain is p -> q[s] -> q[p] -> ... -> q[c] -> p.
        # Total 7 points. So the chain of q's has length 6.
        # s -> p -> ... -> c.
        # The code says:
        # dp_prev[s, s, s+1:] = 1 (Length 2 chain: s, s+1? No, s is start, s is end? No.)
        # Let's trace:
        # dp_prev[s, s, c] = 1 if turn p->s->c is convex.
        # This represents a chain of 2 points from pivot: s, c.
        # Then for l=3 (3 points from pivot: s, p, c):
        # dp_curr[s, p, c] = sum_{k < p} dp_prev[s, k, p] * turn(k, p, c)
        # This builds chains of length l.
        # We need chains of length 6 (s, ..., c).
        # Then close the loop: turn(c, p, s) and turn(p, c, s)?
        # The code checks:
        # cross1: turn q[c] -> p -> q[s] ? No.
        # v1 = q[c] - q[p] ? No, v1x = q[c]-q[p]. v2x = p-q[c].
        # cross1 is turn q[p] -> q[c] -> p.
        # cross2 is turn q[c] -> p -> q[s].
        
        # So we need dp for length 6.
        # The loop `for l in range(3, 7)` updates dp to length 3, 4, 5, 6.
        # Initial dp is length 2.
        
        # Optimization:
        # Use NumPy for the matrix multiplication in the DP step.
        
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for b in range(1, m-1):
            # a in 0..b-1, c in b+1..m-1
            a_indices = np.arange(b)
            c_indices = np.arange(b+1, m)
            
            # q[a] shape (b, 2), q[b] shape (2,), q[c] shape (m-b-1, 2)
            # v1 = q[b] - q[a] (b, 2)
            # v2 = q[c] - q[b] (m-b-1, 2)
            # We need cross(v1_i, v2_j) for all i, j.
            
            # v1_x, v1_y
            v1x = q[b, 0] - q[a_indices, 0]
            v1y = q[b, 1] - q[a_indices, 1]
            
            v2x = q[c_indices, 0] - q[b, 0]
            v2y = q[c_indices, 1] - q[b, 1]
            
            # Cross product: v1x[:, None] * v2y[None, :] - v1y[:, None] * v2x[None, :]
            cross = v1x[:, None] * v2y[None, :] - v1y[:, None] * v2x[None, :]
            
            # Store in turn_mat[a_indices[:, None], b, c_indices[None, :]]
            # This is tricky with advanced indexing.
            # turn_mat[a_indices[:, None], b, c_indices[None, :]] = cross
            # This works if we broadcast indices.
            turn_mat[a_indices[:, None], b, c_indices[None, :]] = cross

        # Precompute turn from pivot p
        # turn_p[s, c] > 0 iff turn p -> q[s] -> q[c] is convex
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m-1):
            c_indices = np.arange(s+1, m)
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[c_indices, 0] - q[s, 0]
            v2y = q[c_indices, 1] - q[s, 1]
            turn_p[s, c_indices] = v1x * v2y - v1y * v2x

        # DP
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        
        # Init length 2
        for s in range(m-1):
            valid_c = np.where(turn_p[s, s+1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for p_idx in range(1, m-1):
                # s < p_idx < c
                s_indices = np.arange(p_idx)
                c_indices = np.arange(p_idx+1, m)
                
                if len(s_indices) == 0 or len(c_indices) == 0:
                    continue
                
                # dp_prev[s, k, p_idx] for all s < k < p_idx?
                # No, dp_prev[s, k, p_idx] where k is the previous point.
                # The recurrence is:
                # dp_curr[s, p_idx, c] = sum_{k} dp_prev[s, k, p_idx] * turn(k, p_idx, c)
                # Here k ranges from s to p_idx-1.
                # But dp_prev[s, k, p_idx] is non-zero only if k > s?
                # In the previous step, dp_prev[s, k, c] was set.
                # So for a fixed s, k can be s, s+1, ..., p_idx-1.
                
                # We have a matrix M of shape (p_idx, p_idx) where M[s, k] = dp_prev[s, k, p_idx]
                # We have a matrix T of shape (p_idx, m-p_idx-1) where T[k, c] = turn_mat[k, p_idx, c]
                # We want Result[s, c] = sum_k M[s, k] * T[k, c]
                
                # Extract M
                # dp_prev[s_indices, :, p_idx] gives shape (p_idx, m, 1)? No.
                # dp_prev is (m, m, m).
                # We want dp_prev[s, k, p_idx] for s in 0..p_idx-1, k in 0..p_idx-1.
                # M = dp_prev[s_indices[:, None], np.arange(p_idx)[None, :], p_idx]
                # This gives shape (p_idx, p_idx).
                
                M = dp_prev[s_indices[:, None], np.arange(p_idx)[None, :], p_idx]
                
                # Extract T
                # T[k, c] = turn_mat[k, p_idx, c] for k in 0..p_idx-1, c in p_idx+1..m-1
                # T = turn_mat[np.arange(p_idx)[:, None], p_idx, c_indices[None, :]]
                # Shape (p_idx, m-p_idx-1)
                T = turn_mat[np.arange(p_idx)[:, None], p_idx, c_indices[None, :]]
                
                # Mask T for convexity
                T_mask = (T > 1e-9).astype(np.int64)
                
                # Result = M @ T_mask
                # Shape (p_idx, m-p_idx-1)
                Result = M @ T_mask
                
                # Store in dp_curr[s_indices[:, None], p_idx, c_indices[None, :]]
                dp_curr[s_indices[:, None], p_idx, c_indices[None, :]] = Result
            
            dp_prev = dp_curr

        # Close the loop
        # For each s, p_idx, c_idx with dp_prev[s, p_idx, c_idx] > 0
        # Check turn q[p_idx] -> q[c_idx] -> p and turn q[c_idx] -> p -> q[s]
        
        # Vectorize the closing check?
        # We can loop s, p_idx, c_idx.
        # m is ~32. 32^3 = 32k iterations. This is fast in Python.
        
        for s in range(m-2):
            for p_idx in range(s+1, m-1):
                for c_idx in range(p_idx+1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue
                    
                    # Turn q[p_idx] -> q[c_idx] -> p
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue
                    
                    # Turn q[c_idx] -> p -> q[s]
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= 1e-9:
                        continue
                    
                    total += count

    return total

def _generate_tight_line_initial(rng):
    n = 33
    x = np.linspace(0.05, 0.95, n)
    # Small jitter to avoid collinearity
    # Area ~ dx * dy. dx ~ 0.03. dy ~ 1e-4 -> Area ~ 3e-6 > 1e-9.
    # If dy is too small, we might fail validity if points are too close?
    # Distance squared < 1e-12. dx ~ 0.03, so d2 > 1e-4. Safe.
    y = 0.5 + rng.normal(0, 1e-4, n)
    
    pts = np.column_stack([x, y])
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        # Increase jitter slightly or shift
        y = 0.5 + rng.normal(0, 1e-3, n)
        pts = np.column_stack([x, y])
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
        
    return pts

def _hill_climb_linear(pts, rng, time_limit):
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 50
    step_size = 1e-4
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        # Move a random point's y-coordinate slightly
        # Or x-coordinate? Changing x changes the order, which is costly to re-sort but DP handles it.
        # However, keeping x fixed preserves the "linear" structure better.
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        if rng.random() < 0.8:
            # Perturb y
            new_pts[i, 1] += rng.normal(0, step_size)
        else:
            # Perturb x (careful with order)
            new_pts[i, 0] += rng.normal(0, step_size * 0.1)
            
        new_pts = np.clip(new_pts, 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                no_improve = 0
                step_size *= 0.9
            else:
                no_improve += 1
                if no_improve % 10 == 0:
                    step_size = min(1e-2, step_size * 1.5)
        else:
            no_improve += 1

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
    
    # Initialize with tight line
    best_pts = _generate_tight_line_initial(rng)
    best_c = _count_convex7_fast(best_pts)
    
    # Hill climb
    if time_left() > 10:
        pts, c = _hill_climb_linear(best_pts, rng, time_left())
        if c < best_c:
            best_c = c
            best_pts = pts
    
    return best_pts
