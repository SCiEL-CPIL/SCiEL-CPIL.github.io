"""Circle packing in perimeter-4 rectangle: SLSQP + Hexagonal Init + Soft Repulsion + Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

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
        if i == i: 
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)

def _soft_repulsion_relax(x, y, r, w, h, time_budget, alpha=0.1, beta=0.5):
    """
    Soft repulsion gradient descent that pushes circles apart even when not overlapping.
    Uses a smooth potential based on gap = r_i + r_j - d.
    """
    x = x.copy()
    y = y.copy()
    n = N
    dt = 0.002
    min_dt = 0.00005
    max_iter = int(min(3000, time_budget * 150))
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion (smooth)
        eps = 1e-6
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100
        
        # Pairwise soft repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        # Gap = r_sum - d; positive means overlap
        gap = r_sum_pairs - d_pairs
        # Soft force: f = alpha * gap + beta * gap^3 (stronger when overlapping)
        # Only apply when gap > -0.1 (within interaction range)
        mask = gap > -0.1
        if not np.any(mask):
            break
            
        f_pairs = np.zeros(len(iu[0]))
        f_pairs[mask] = alpha * gap[mask] + beta * gap[mask]**3
        # Clamp force
        f_pairs = np.clip(f_pairs, -1.0, 1.0)
        
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
            
        # Adaptive step
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 0.3:
            dt = max(min_dt, dt * 0.7)
        elif force_mag < 0.01:
            dt = min(0.005, dt * 1.1)
            
        x += dt * fx
        y += dt * fy
        
        # Clamp to bounds
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    return x, y

def _expand_and_relax(x, y, r, w, h, time_budget):
    """
    Iteratively expand each circle to its max possible radius, then relax positions.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    max_rounds = 80
    
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        
        # Step 1: Expand each circle
        for i in range(n):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
        
        # Step 2: Relax positions
        x, y = _soft_repulsion_relax(x, y, r, w, h, time_budget - 0.3)
        
        if not improved:
            # Try one more pass
            for i in range(n):
                r_max = _max_radius_for_circle(x, y, r, w, h, i)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
                
    return x, y, r

def _hexagonal_init(w, h, spacing):
    """Initialize circles in a hexagonal lattice pattern."""
    x = []
    y = []
    r = []
    
    dy = spacing * np.sqrt(3) / 2
    row = 0
    yy = spacing / 2
    while yy < h:
        offset = 0 if row % 2 == 0 else spacing / 2
        xx = spacing / 2 + offset
        while xx < w:
            x.append(xx)
            y.append(yy)
            r.append(spacing / 2 * 0.95)
            xx += spacing
        yy += dy
        row += 1
    
    # Trim to N circles
    if len(x) > N:
        x = x[:N]
        y = y[:N]
        r = r[:N]
    elif len(x) < N:
        # Add remaining circles in random positions
        rng = np.random.default_rng(123)
        while len(x) < N:
            r0 = rng.uniform(0.04, 0.08)
            cx = rng.uniform(r0, w - r0)
            cy = rng.uniform(r0, h - r0)
            x.append(cx)
            y.append(cy)
            r.append(r0)
    
    return np.array(x), np.array(y), np.array(r)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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

    def get_initial_guesses():
        guesses = []
        
        # Hexagonal lattice inits with different aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            # Estimate spacing for hex packing
            area_per_circle = w0 * h0 / N
            spacing = np.sqrt(area_per_circle / (np.sqrt(3)/4))
            x, y, r = _hexagonal_init(w0, h0, spacing)
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = x
            z[1::3][:N] = y
            z[2::3][:N] = r
            z[-1] = w0
            guesses.append(z)
            
        # Grid-based inits
        side = int(np.ceil(np.sqrt(N)))
        for w0 in [1.0, 0.85, 1.18, 0.75, 1.33]:
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
            
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    guesses = get_initial_guesses()
    
    # Phase 1: SLSQP Optimization
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break
            
        x0 = z0[0::3][:N].copy()
        y0 = z0[1::3][:N].copy()
        r0 = z0[2::3][:N].copy()
        w0 = z0[-1]
        h0 = 2.0 - w0
        
        # Force relax to resolve initial overlaps
        x_rel, y_rel = _soft_repulsion_relax(x0, y0, r0, w0, h0, time.time() + 0.8)
        
        z_relaxed = np.zeros(3 * N + 1)
        z_relaxed[0::3][:N] = x_rel
        z_relaxed[1::3][:N] = y_rel
        z_relaxed[2::3][:N] = r0
        z_relaxed[-1] = w0
        
        # SLSQP with penalty
        penalty_weight = 1000.0
        
        def objective(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            
            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6
                
            obj = -np.sum(r)
            
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
            
            obj += overlap_penalty + boundary_penalty
            
            if w < 0.5:
                obj += penalty_weight * (0.5 - w) ** 2
            elif w > 1.5:
                obj += penalty_weight * (w - 1.5) ** 2
                
            return obj

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(80, remaining * 15)))
        
        try:
            res = minimize(objective, z_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
                    
        except Exception:
            continue
            
    # Phase 2: Polish with expansion and relaxation
    if best_result is not None and time.time() < deadline - 15:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        time_budget = deadline - 5
        x, y, r = _expand_and_relax(x, y, r, w, h, time_budget)
        
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z_new = np.zeros(3 * N + 1)
                z_new[0::3][:N] = x
                z_new[1::3][:N] = y
                z_new[2::3][:N] = r
                z_new[-1] = w
                best_result = z_new

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
