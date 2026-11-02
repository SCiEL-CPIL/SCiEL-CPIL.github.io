"""Circle packing in perimeter-4 rectangle: Optimized SLSQP + Exact Local Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_NUM_PAIRS = N * (N - 1) // 2
_IU = np.triu_indices(N, k=1)


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    d_sq_pairs = d_sq[_IU]
    r_sum_pairs = r_sum[_IU]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid. Vectorized for speed."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    
    # Pairwise check
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Only check upper triangle
    valid_pairs = True
    # Using a loop is often faster for N=21 than creating a full mask if we just need a boolean
    # But let's stick to vectorized for consistency, checking all pairs is fine for N=21
    # Actually, a simple loop is very fast for N=21 and avoids memory allocation overhead of masks
    for i in range(N):
        for j in range(i + 1, N):
            if d_sq[i, j] < (r_sum[i, j] ** 2) - tol:
                return False
    return True


def _max_radius_for_circle(x, y, r, w, h, i, tol=1e-10):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)

def _is_valid_move(x, y, r, w, h, i, new_x, new_y, tol=1e-10):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] - tol or new_x > w - r[i] + tol:
        return False
    if new_y < r[i] - tol or new_y > h - r[i] + tol:
        return False
    
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - tol:
            return False
    return True


def _exact_local_expansion_polish(x, y, r, w, h, time_budget):
    """
    Advanced polish: Iteratively expand circles to their maximum possible size.
    If blocked, attempt to move the blocking circle or the current circle to free space.
    This is an exact local search, not a gradient-based one.
    """
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # We will iterate until we can't improve anymore or time runs out
    # Each "round" we try to expand every circle as much as possible
    max_rounds = 50
    
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
            
        improved = False
        
        # Try to expand each circle
        for i in range(n):
            # 1. Determine current max radius
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            
            if r_max > r[i] + 1e-8:
                # We can expand. Binary search for the precise limit if needed, 
                # but r_max is already exact given fixed positions.
                # However, expanding might violate constraints if we just set it?
                # No, r_max is calculated based on current positions of others.
                # So setting r[i] = r_max is valid.
                r[i] = r_max
                improved = True
            else:
                # We are blocked. Try to create space.
                # Identify the blocker
                min_dist = float('inf')
                blocker_type = None
                blocker_idx = -1
                blocker_vec = np.array([0.0, 0.0])
                
                # Check walls
                d_left = x[i] - r[i]
                d_right = w - x[i] - r[i]
                d_bottom = y[i] - r[i]
                d_top = h - y[i] - r[i]
                
                if d_left < min_dist:
                    min_dist = d_left
                    blocker_type = 'wall'
                    blocker_idx = 0 # left
                    blocker_vec = np.array([-1.0, 0.0])
                if d_right < min_dist:
                    min_dist = d_right
                    blocker_type = 'wall'
                    blocker_idx = 1 # right
                    blocker_vec = np.array([1.0, 0.0])
                if d_bottom < min_dist:
                    min_dist = d_bottom
                    blocker_type = 'wall'
                    blocker_idx = 2 # bottom
                    blocker_vec = np.array([0.0, -1.0])
                if d_top < min_dist:
                    min_dist = d_top
                    blocker_type = 'wall'
                    blocker_idx = 3 # top
                    blocker_vec = np.array([0.0, 1.0])
                
                # Check circles
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_dist:
                        min_dist = gap
                        blocker_type = 'circle'
                        blocker_idx = j
                        if d > 1e-9:
                            blocker_vec = np.array([dx / d, dy / d])
                        else:
                            blocker_vec = np.array([1.0, 0.0]) # Arbitrary if coincident
                
                # If we are very close to a blocker (gap < epsilon), try to move
                if min_dist < 0.01:
                    # Try to move circle i away from blocker
                    # Or move blocker away from i
                    # Moving i is safer as we just re-check validity
                    
                    # Try small steps in the direction away from blocker
                    # And also perpendicular to slide around
                    move_dirs = []
                    move_dirs.append(blocker_vec) # Direct away
                    move_dirs.append(np.array([-blocker_vec[1], blocker_vec[0]])) # Perp 1
                    move_dirs.append(np.array([blocker_vec[1], -blocker_vec[0]])) # Perp 2
                    
                    moved = False
                    for dir_vec in move_dirs:
                        if np.linalg.norm(dir_vec) < 1e-6:
                            continue
                        # Try a few step sizes
                        for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                            new_xi = x[i] + dir_vec[0] * step
                            new_yi = y[i] + dir_vec[1] * step
                            if _is_valid_move(x, y, r, w, h, i, new_xi, new_yi):
                                x[i] = new_xi
                                y[i] = new_yi
                                improved = True
                                moved = True
                                break
                        if moved:
                            break
                    
                    # If we moved i, we might be able to expand it now or others.
                    # Continue to next circle in this round.
                    
                    # Alternative: Try to move the blocker (if it's a circle)
                    # This is complex to implement safely without breaking other constraints.
                    # The "move i" strategy is robust.
        
        if not improved:
            break
            
    # Final global squeeze
    # Try to scale all radii up slightly
    # Binary search for scaling factor
    lo = 1.0
    hi = 1.05
    for _ in range(30):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
            r = r_trial
        else:
            hi = mid
            
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        if not _is_valid(x, y, r, w, h, tol):
            return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # 1. Grid-based aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # 2. Hexagonal packing (Square container)
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # 3. Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.7, 1.3)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.06, 0.1)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Optimization Phase
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Quick force relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_fast(z0, max_iter=100)
        
        # SLSQP Optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 30)))
        
        try:
            res = minimize(_vectorized_objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
                
        except Exception:
            pass

    # Polish Phase
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
        w = best_result[-1]
        h = 2.0 - w
        
        # Run the exact local expansion polish
        x_pol, y_pol, r_pol = _exact_local_expansion_polish(x, y, r, w, h, deadline - 5)
        
        if _is_valid(x_pol, y_pol, r_pol, w, h):
            val_pol = float(np.sum(r_pol))
            if val_pol > best_value:
                best_value = val_pol
                best_result = np.zeros(3 * N + 1)
                best_result[0::3][:N] = x_pol
                best_result[1::3][:N] = y_pol
                best_result[2::3][:N] = r_pol
                best_result[-1] = w

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety check and formatting
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    # Tiny shrink to ensure strict validity if needed
    for scale in [1.0, 0.999999, 0.99999]:
        if _is_valid(x, y, r * scale, w, h, tol=1e-9):
            r = r * scale
            break
            
    return np.column_stack((x, y, r))


def _force_relax_fast(z, max_iter=100):
    """Fast force-directed relaxation to remove initial overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.01
    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        fx -= np.maximum(r - x, 0.0) * 10
        fx += np.maximum(x + r - w, 0.0) * 10
        fy -= np.maximum(r - y, 0.0) * 10
        fy += np.maximum(y + r - h, 0.0) * 10
        
        # Pair repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(r_sum - d, 0.0)
        
        # Force is proportional to overlap
        # F_ij = overlap * (x_i - x_j) / d
        # Sum over j for each i
        # Vectorized sum is tricky due to division by zero, so we mask
        mask = overlap > 0
        d_safe = np.where(mask, d, 1.0)
        
        fx_vec = (overlap * dx / d_safe)
        fy_vec = (overlap * dy / d_safe)
        
        # Sum over neighbors (axis 1)
        fx += np.sum(fx_vec, axis=1)
        fy += np.sum(fy_vec, axis=1)
        
        # Update
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        
        # Clamp
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
            
        # Check convergence
        if np.max(overlap) < 1e-6:
            break
            
    return z
