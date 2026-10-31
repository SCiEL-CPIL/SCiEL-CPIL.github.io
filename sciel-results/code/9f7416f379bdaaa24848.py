"""Circle packing in perimeter-4 rectangle: Multi-start grid + L-BFGS-B polish + per-circle expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=10000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty (cubic for stronger gradient)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty (cubic)
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
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_radius_for_circle(x, y, r, w, h, i):
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _expand_all(x, y, r, w, h, max_rounds=50, time_budget=None):
    """Iteratively expand each circle to its max possible radius with position adjustment."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_rounds):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
        
        # Try to shift circles to create more space
        if improved:
            for i in range(N):
                # Check if circle is touching constraints
                min_margin = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
                if min_margin < 1e-6:
                    # Try small random perturbations
                    for _ in range(5):
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        new_x = x[i] + dx
                        new_y = y[i] + dy
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            break
    
        if not improved:
            break
    return x, y, r


def _force_relax(z, max_iter=500):
    """Force-directed relaxation with adaptive step size and damping."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.00001
    damping = 0.95

    vx = np.zeros(n)
    vy = np.zeros(n)

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (stronger)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-8:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < 1e-8 and np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break

        # Adaptive step size with velocity damping
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 2.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.05:
            dt = min(0.01, dt * 1.2)

        # Update velocities with damping
        vx = vx * damping + dt * fx
        vy = vy * damping + dt * fy

        # Update positions
        z[0::3][:n] += vx
        z[1::3][:n] += vy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _get_initial_guesses():
    """Generate diverse initial guesses using grid layouts, corner packing, and random starts."""
    guesses = []
    rng = np.random.default_rng(42)
    
    # Grid layouts with varying aspect ratios and cell sizes
    for w0 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Try different grid configurations
        for cols in [3, 4, 5]:
            rows = int(np.ceil(N / cols))
            if rows * cols < N:
                continue
            
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, cols)
                if row >= rows:
                    break
                cx = (col + 0.5) / cols * w0
                cy = (row + 0.5) / rows * h0
                r0 = min(w0 / cols, h0 / rows) * 0.42
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
    
    # Corner-based packing: place large circles in corners, fill center
    for w0 in [0.8, 1.0, 1.2]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        corner_r = min(w0, h0) * 0.25
        positions = [
            (corner_r, corner_r),
            (w0 - corner_r, corner_r),
            (corner_r, h0 - corner_r),
            (w0 - corner_r, h0 - corner_r),
        ]
        for i in range(min(4, N)):
            z[3 * i] = positions[i][0]
            z[3 * i + 1] = positions[i][1]
            z[3 * i + 2] = corner_r
        
        # Fill remaining with smaller circles in center region
        cx_center = w0 / 2.0
        cy_center = h0 / 2.0
        small_r = min(w0, h0) * 0.08
        for i in range(4, N):
            row = (i - 4) // 4
            col = (i - 4) % 4
            cx = cx_center + (col - 1.5) * small_r * 2.2
            cy = cy_center + (row - 1.0) * small_r * 2.2
            if cx - small_r < 0 or cx + small_r > w0 or cy - small_r < 0 or cy + small_r > h0:
                cx = cx_center + rng.uniform(-0.2, 0.2)
                cy = cy_center + rng.uniform(-0.2, 0.2)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = small_r
        z[-1] = w0
        guesses.append(z)
    
    # Random starts with varied radii distributions
    for trial in range(25):
        w0 = rng.uniform(0.5, 1.5)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        
        # Vary the radius distribution
        base_r = rng.uniform(0.04, 0.10)
        for i in range(N):
            r0 = base_r * rng.uniform(0.7, 1.3)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)
    
    # Hexagonal packing initializations
    for w0 in [0.8, 1.0, 1.2]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        spacing = min(w0, h0) / 5.0
        r0 = spacing * 0.45
        row = 0
        col = 0
        for i in range(N):
            cx = (col + 0.5) * spacing + (row % 2) * spacing * 0.5
            cy = (row + 0.5) * spacing * 0.866
            if cx + r0 > w0 or cy + r0 > h0:
                col = 0
                row += 1
                cx = (col + 0.5) * spacing + (row % 2) * spacing * 0.5
                cy = (row + 0.5) * spacing * 0.866
                if cx + r0 > w0 or cy + r0 > h0:
                    break
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
            col += 1
        z[-1] = w0
        guesses.append(z)
    
    # Ring/circular packing: circles arranged in concentric rings
    for w0 in [0.9, 1.0, 1.1]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        cx_center = w0 / 2.0
        cy_center = h0 / 2.0
        max_r = min(w0, h0) * 0.45
        
        # Center circle
        center_r = max_r * 0.3
        z[0] = cx_center
        z[1] = cy_center
        z[2] = center_r
        
        # Outer ring
        outer_ring_r = max_r * 0.2
        n_outer = N - 1
        for i in range(n_outer):
            angle = 2 * np.pi * i / n_outer
            cx = cx_center + (center_r + outer_ring_r) * np.cos(angle)
            cy = cy_center + (center_r + outer_ring_r) * np.sin(angle)
            if cx - outer_ring_r < 0 or cx + outer_ring_r > w0 or cy - outer_ring_r < 0 or cy + outer_ring_r > h0:
                cx = cx_center + rng.uniform(-0.3, 0.3)
                cy = cy_center + rng.uniform(-0.3, 0.3)
            z[3 * (i + 1)] = cx
            z[3 * (i + 1) + 1] = cy
            z[3 * (i + 1) + 2] = outer_ring_r
        z[-1] = w0
        guesses.append(z)
    
    return guesses


def _polish_with_lbfgsb(z, time_budget):
    """Polish using L-BFGS-B on the penalized objective with multi-stage penalty."""
    z = z.copy().astype(float)
    n = N
    
    # Multi-stage: start with lower penalty, increase gradually
    for stage, pw in enumerate([1000.0, 5000.0, 20000.0]):
        if time.time() > time_budget:
            break
        
        def objective(zv):
            return _vectorized_objective(zv, penalty_weight=pw)
        
        remaining = time_budget - time.time()
        if remaining < 2:
            break
        
        maxiter = int(min(150, max(20, remaining * 15)))
        try:
            res = minimize(objective, z, method='L-BFGS-B',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            z = res.x
        except Exception:
            break
    
    return z


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # 1. Try to expand radius i to max possible using binary search
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                # Binary search for the true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True

            # 2. If expansion is limited, try to shift circle i to create space
            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]

            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j

            # If touching a constraint, try multi-step moves away
            if min_dist < 0.02:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d

                # Try multiple step sizes, from larger to smaller
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                else:
                    # Try perpendicular directions too
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    # Final uniform expansion with binary search
    grow_factor = 1.001
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _simulated_annealing(z, time_budget, max_iters=5000):
    """Simulated annealing with position and radius moves."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    rng = np.random.default_rng(123)
    
    def get_energy(x, y, r, w, h):
        """Compute energy: sum of overlaps + boundary violations."""
        energy = 0.0
        # Boundary violations
        for i in range(n):
            if x[i] < r[i]:
                energy += (r[i] - x[i]) ** 2 * 100
            if x[i] > w - r[i]:
                energy += (x[i] - (w - r[i])) ** 2 * 100
            if y[i] < r[i]:
                energy += (r[i] - y[i]) ** 2 * 100
            if y[i] > h - r[i]:
                energy += (y[i] - (h - r[i])) ** 2 * 100
        # Pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    energy += (r_sum - d) ** 2 * 50
        return energy
    
    def is_valid_config(x, y, r, w, h, tol=1e-9):
        for i in range(n):
            if r[i] <= 0:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False
        return True
    
    T = 0.1
    T_min = 0.0001
    cooling = 0.9995
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_val = -np.sum(r) if is_valid_config(x, y, r, w, h) else -1e6
    
    start_t = time.time()
    for it in range(max_iters):
        if time.time() > time_budget:
            break
        
        # Choose move type
        move_type = rng.random()
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.5:
            # Move position of one circle
            i = rng.integers(n)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            new_x[i] = np.clip(x[i] + dx, r[i], w - r[i])
            new_y[i] = np.clip(y[i] + dy, r[i], h - r[i])
        elif move_type < 0.8:
            # Adjust radius of one circle
            i = rng.integers(n)
            factor = rng.uniform(0.95, 1.05)
            new_r[i] = np.clip(r[i] * factor, 0.001, min(x[i], w - x[i], y[i], h - y[i]))
        else:
            # Swap two circles
            i, j = rng.choice(n, size=2, replace=False)
            new_x[[i, j]] = x[[j, i]]
            new_y[[i, j]] = y[[j, i]]
            new_r[[i, j]] = r[[j, i]]
        
        # Compute energy difference
        old_energy = get_energy(x, y, r, w, h)
        new_energy = get_energy(new_x, new_y, new_r, w, h)
        delta_e = new_energy - old_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            x, y, r = new_x, new_y, new_r
            T *= cooling
            
            # Check if this is a valid configuration
            if is_valid_config(x, y, r, w, h):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
        
        # Periodically try to expand all radii
        if it % 100 == 99:
            for i in range(n):
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i]:
                    r[i] = r_max
    
    # Return best valid configuration
    if is_valid_config(best_x, best_y, best_r, w, h):
        z_out = np.empty(3 * N + 1)
        z_out[0::3][:n] = best_x
        z_out[1::3][:n] = best_y
        z_out[2::3][:n] = best_r
        z_out[-1] = w
        return z_out
    return None


def _global_scale_expansion(x, y, r, w, h, time_budget):
    """Aggressively scale all radii up while maintaining validity."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # First, do per-circle max expansion
    for _ in range(20):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(N):
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
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
        if not improved:
            break
    
    # Then, try uniform scaling
    scale = 1.0
    for _ in range(200):
        if time.time() > time_budget:
            break
        trial_r = r * (scale + 0.001)
        if _is_valid(x, y, trial_r, w, h):
            scale += 0.001
            r = trial_r
        else:
            break
    
    # Try to shift circles to allow more scaling
    for _ in range(50):
        if time.time() > time_budget:
            break
        # Find the most constrained circle
        min_gap = float('inf')
        min_idx = 0
        for i in range(N):
            gap = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = min(gap, d - r[i] - r[j])
            if gap < min_gap:
                min_gap = gap
                min_idx = i
        
        if min_gap > 0.01:
            break
        
        # Try to move the most constrained circle
        i = min_idx
        # Determine direction to move
        dx_left = x[i] - r[i]
        dx_right = w - x[i] - r[i]
        dy_bottom = y[i] - r[i]
        dy_top = h - y[i] - r[i]
        
        # Find nearest neighbor
        nearest_j = -1
        nearest_d = float('inf')
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < nearest_d:
                nearest_d = d
                nearest_j = j
        
        if nearest_j >= 0 and nearest_d - r[i] - r[nearest_j] < 0.01:
            # Move away from nearest neighbor
            dx = x[i] - x[nearest_j]
            dy = y[i] - y[nearest_j]
            d = np.sqrt(dx * dx + dy * dy)
            if d > 1e-6:
                ux = dx / d
                uy = dy / d
                for step in [0.02, 0.01, 0.005]:
                    new_x = x[i] + ux * step
                    new_y = y[i] + uy * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        break
        else:
            # Move toward center if touching wall
            if dx_left < 0.01:
                x[i] += 0.01
            elif dx_right < 0.01:
                x[i] -= 0.01
            elif dy_bottom < 0.01:
                y[i] += 0.01
            elif dy_top < 0.01:
                y[i] -= 0.01
        
        # Try to expand again
        for i in range(N):
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
            r_max = max(0.0, r_max)
            if r_max > r[i]:
                r[i] = r_max
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

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

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    guesses = _get_initial_guesses()
    
    # Sort guesses by initial quality (rough estimate)
    def initial_quality(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        # Check for obvious violations
        violations = 0
        for i in range(N):
            if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                violations += 1
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2:
                    violations += 1
        
        return -violations * 10 + np.sum(r)
    
    guesses.sort(key=initial_quality, reverse=True)

    # Process top guesses with multiple optimizers
    num_to_process = min(len(guesses), 12)
    
    for i, z0 in enumerate(guesses[:num_to_process]):
        if time.time() > deadline - 20:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=300)

        # Try L-BFGS-B
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break

        z_opt = _polish_with_lbfgsb(z0_relaxed, deadline - 15)

        # Expand radii
        x = z_opt[0::3][:N]
        y = z_opt[1::3][:N]
        r = z_opt[2::3][:N]
        w = z_opt[-1]
        h = 2.0 - w

        x, y, r = _expand_all(x, y, r, w, h, max_rounds=30, time_budget=deadline - 15)

        # Reconstruct z
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w

        valid, val = check_validity(z_new)
        if valid and val > best_value:
            best_value = val
            best_result = z_new.copy()
        
        # Also try simulated annealing on a subset of guesses
        if i < 6 and time.time() < deadline - 30:
            z_sa = _simulated_annealing(z0_relaxed, deadline - 25, max_iters=3000)
            if z_sa is not None:
                valid_sa, val_sa = check_validity(z_sa)
                if valid_sa and val_sa > best_value:
                    best_value = val_sa
                    best_result = z_sa.copy()

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 20:
        # 1. Global scale expansion
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        x, y, r = _global_scale_expansion(x, y, r, w, h, deadline - 15)
        
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w
        
        valid, val = check_validity(z_new)
        if valid and val > best_value:
            best_value = val
            best_result = z_new.copy()

        # 2. Expansion/Relocation Polish
        if time.time() < deadline - 12:
            expanded = _expansion_relaxation_polish(best_result, deadline - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Final L-BFGS-B polish
        if best_result is not None and time.time() < deadline - 8:
            z_opt = _polish_with_lbfgsb(best_result, deadline - 5)
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w
            x, y, r = _expand_all(x, y, r, w, h, max_rounds=30, time_budget=deadline - 5)
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:N] = x
            z_new[1::3][:N] = y
            z_new[2::3][:N] = r
            z_new[-1] = w
            valid, val = check_validity(z_new)
            if valid and val > best_value:
                best_value = val
                best_result = z_new.copy()
        
        # 4. Final simulated annealing polish
        if best_result is not None and time.time() < deadline - 5:
            z_final = _simulated_annealing(best_result, deadline - 3, max_iters=2000)
            if z_final is not None:
                valid, val = check_validity(z_final)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_final.copy()

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
