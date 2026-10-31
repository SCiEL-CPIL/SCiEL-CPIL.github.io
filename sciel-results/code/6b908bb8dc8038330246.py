"""Circle packing in perimeter-4 rectangle: Force-directed soft-constraint optimization."""
import numpy as np
import time

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    if W <= 0.01 or H <= 0.01:
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

def _soft_penalty(z, W, H):
    """
    z layout: [x(21), y(21), r(21)]
    Returns (penalty, negative_sum_r) so we minimize penalty - sum_r.
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:]
    
    penalty = 0.0
    
    # Wall penalties (quadratic)
    wall_pen = 0.0
    for i in range(N):
        if x[i] - r[i] < 0:
            wall_pen += (x[i] - r[i]) ** 2
        if x[i] + r[i] > W:
            wall_pen += (x[i] + r[i] - W) ** 2
        if y[i] - r[i] < 0:
            wall_pen += (y[i] - r[i]) ** 2
        if y[i] + r[i] > H:
            wall_pen += (y[i] + r[i] - H) ** 2
    penalty += 10.0 * wall_pen
    
    # Pairwise overlap penalties (quadratic)
    pair_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if d < min_d:
                pair_pen += (min_d - d) ** 2
    penalty += 10.0 * pair_pen
    
    # Radius lower bound
    if np.any(r < 0.005):
        penalty += 100.0 * np.sum(np.maximum(0, 0.005 - r)) ** 2
    
    # Perimeter penalty (soft)
    perim = 2.0 * (W + H)
    if perim > 4.0:
        penalty += 50.0 * (perim - 4.0) ** 2
    
    # Aspect ratio penalty (encourage reasonable aspect)
    if W > 0.1 and H > 0.1:
        ar = W / H
        if ar > 2.0 or ar < 0.5:
            penalty += 5.0 * (max(0, ar - 2.0) + max(0, 0.5 - ar)) ** 2
    
    obj = penalty - np.sum(r)
    return obj

def _gradient(z, W, H):
    """Numerical gradient of _soft_penalty."""
    eps = 1e-6
    grad = np.zeros_like(z)
    base = _soft_penalty(z, W, H)
    for i in range(len(z)):
        z_p = z.copy()
        z_p[i] += eps
        grad[i] = (_soft_penalty(z_p, W, H) - base) / eps
    return grad

def _fd_gradient_full(z, W, H):
    """Full gradient including W,H optimization."""
    # z layout: [x(21), y(21), r(21), W, H]
    eps = 1e-5
    grad = np.zeros_like(z)
    base = _soft_penalty(z[:3*N], W, H)
    
    for i in range(3 * N):
        z_p = z.copy()
        z_p[i] += eps
        grad[i] = (_soft_penalty(z_p[:3*N], z_p[3*N], z_p[3*N+1]) - base) / eps
    
    # W gradient
    z_p = z.copy()
    z_p[3*N] += eps
    grad[3*N] = (_soft_penalty(z_p[:3*N], z_p[3*N], z_p[3*N+1]) - base) / eps
    
    # H gradient  
    z_p = z.copy()
    z_p[3*N+1] += eps
    grad[3*N+1] = (_soft_penalty(z_p[:3*N], z_p[3*N], z_p[3*N+1]) - base) / eps
    
    return grad

def _force_directed_solve(W0, H0, seed, max_iter=2000, lr0=0.05, lr_decay=0.999):
    """
    Force-directed optimization with gradient descent.
    Returns (x, y, r, W, H).
    """
    rng = np.random.default_rng(seed)
    
    # Initialize
    x = rng.uniform(0.1 * W0, 0.9 * W0, N)
    y = rng.uniform(0.1 * H0, 0.9 * H0, N)
    r = np.full(N, 0.12)
    W = W0
    H = H0
    
    z = np.concatenate([x, y, r, [W, H]])
    
    lr = lr0
    best_obj = _soft_penalty(z[:3*N], z[3*N], z[3*N+1])
    best_z = z.copy()
    patience = 0
    max_patience = 200
    
    for it in range(max_iter):
        # Compute gradient
        grad = _fd_gradient_full(z, z[3*N], z[3*N+1])
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr = lr0 * (lr_decay ** (it // 50))
        
        # Gradient step with momentum-like damping
        z_new = z - lr * grad
        
        # Enforce constraints on z_new
        # W + H = 2 (perimeter = 4)
        W_new = z_new[3*N]
        H_new = z_new[3*N+1]
        # Project to W + H = 2
        s = W_new + H_new
        if s > 1e-6:
            W_new = 2.0 * W_new / s
            H_new = 2.0 * H_new / s
        z_new[3*N] = W_new
        z_new[3*N+1] = H_new
        
        # Clamp positions and radii
        z_new[:N] = np.clip(z_new[:N], 0.01, max(0.02, W_new - 0.01))
        z_new[N:2*N] = np.clip(z_new[N:2*N], 0.01, max(0.02, H_new - 0.01))
        z_new[2*N:3*N] = np.clip(z_new[2*N:3*N], 0.005, min(W_new, H_new) / 2.0)
        z_new[3*N] = np.clip(z_new[3*N], 0.3, 1.7)
        z_new[3*N+1] = np.clip(z_new[3*N+1], 0.3, 1.7)
        # Re-project after clipping
        s = z_new[3*N] + z_new[3*N+1]
        if s > 1e-6:
            z_new[3*N] = 2.0 * z_new[3*N] / s
            z_new[3*N+1] = 2.0 * z_new[3*N+1] / s
        
        obj_new = _soft_penalty(z_new[:3*N], z_new[3*N], z_new[3*N+1])
        
        if obj_new < best_obj - 1e-10:
            best_obj = obj_new
            best_z = z_new.copy()
            patience = 0
        else:
            patience += 1
            # If stuck, add jitter
            if patience > 50:
                jitter = 0.01 * rng.standard_normal(len(z))
                z = z_new + jitter
                z[3*N] = best_z[3*N]
                z[3*N+1] = best_z[3*N+1]
                patience = 0
            else:
                z = z_new
        
        if patience > max_patience:
            break
        
        # Convergence check
        if np.max(np.abs(grad)) < 1e-8:
            break
    
    return best_z[:N], best_z[N:2*N], best_z[2*N:3*N], best_z[3*N], best_z[3*N+1]

def _expand_radii(x, y, r, W, H, max_iter=100):
    """
    Bisection-based radius expansion: try to grow each circle as much as possible.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Binary search for max radius of circle i given others fixed
            lo = r[i]
            hi = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                hi = min(hi, d - r[j])
            
            if hi <= lo + 1e-9:
                continue
            
            # Bisection
            for _b in range(30):
                mid = (lo + hi) / 2
                # Check if circle i with radius mid is valid
                valid = True
                if x[i] - mid < 0 or x[i] + mid > W or y[i] - mid < 0 or y[i] + mid > H:
                    valid = False
                else:
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < mid + r[j] - 1e-10:
                            valid = False
                            break
                
                if valid:
                    lo = mid
                else:
                    hi = mid
            
            if lo > r[i] + 1e-8:
                r[i] = lo
                improved = True
        
        if not improved:
            break
    
    return x, y, r

def _local_search_positions(x, y, r, W, H, max_iter=50):
    """
    Try to reposition circles to allow further radius growth.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Current max radius
            r_max_cur = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_max_cur = min(r_max_cur, d - r[j])
            
            # Try 8 directions
            for dx_m in [-0.02, 0, 0.02]:
                for dy_m in [-0.02, 0, 0.02]:
                    if dx_m == 0 and dy_m == 0:
                        continue
                    x_new = x[i] + dx_m
                    y_new = y[i] + dy_m
                    
                    # Check walls
                    if x_new - r[i] < 0 or x_new + r[i] > W:
                        continue
                    if y_new - r[i] < 0 or y_new + r[i] > H:
                        continue
                    
                    # Check neighbors
                    valid = True
                    r_max_new = min(x_new, W - x_new, y_new, H - y_new)
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x_new - x[j]
                        dy = y_new - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < r[i] + r[j] - 1e-10:
                            valid = False
                            break
                        r_max_new = min(r_max_new, d - r[j])
                    
                    if valid and r_max_new > r_max_cur + 1e-4:
                        x[i] = x_new
                        y[i] = y_new
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, W, H):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), W, H)
    
    # Aspect ratios to try: W (with H = 2 - W)
    # Perimeter constraint: 2(W+H) = 4 => W+H = 2
    aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.9, 1.1, 0.8, 1.2, 0.65, 1.35, 0.95, 1.05]
    
    n_aspects = len(aspect_ratios)
    
    for idx, W0 in enumerate(aspect_ratios):
        if time.time() > deadline - 15:
            break
        H0 = 2.0 - W0
        if H0 < 0.3 or W0 < 0.3:
            continue
        
        remaining = n_aspects - idx
        time_per_aspect = (deadline - time.time() - 10) / remaining
        
        # Number of seeds based on available time
        n_seeds = max(1, int(time_per_aspect / 8.0))
        n_seeds = min(n_seeds, 4)
        
        for seed in range(n_seeds):
            if time.time() > deadline - 10:
                break
            
            # Force-directed solve
            x, y, r, W, H = _force_directed_solve(W0, H0, seed * 1000 + idx, max_iter=1500)
            
            # Local position search
            x, y, r = _local_search_positions(x, y, r, W, H)
            
            # Expand radii
            x, y, r = _expand_radii(x, y, r, W, H)
            
            update_best(x, y, r, W, H)
            
            # Second round of expansion after position tweaks
            x, y, r = _local_search_positions(x, y, r, W, H)
            x, y, r = _expand_radii(x, y, r, W, H)
            update_best(x, y, r, W, H)
    
    if best_result is None:
        # Fallback: simple grid
        W, H = 1.0, 1.0
        cols = 5
        rows = 5
        x = []
        y = []
        r = []
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= N:
                    break
                x.append((col + 0.5) * W / cols)
                y.append((row + 0.5) * H / rows)
                r.append(0.08)
                count += 1
        x = np.array(x[:N])
        y = np.array(y[:N])
        r = np.array(r[:N])
        best_result = (x, y, r, W, H)
    
    x, y, r, W, H = best_result
    
    # Final safety: verify and shrink if needed
    for _ in range(10):
        if _is_valid(x, y, r, W, H):
            break
        r = r * 0.999
        if np.max(r) < 0.01:
            # Last resort
            W, H = 1.0, 1.0
            x = np.linspace(0.15, 0.85, N)
            y = np.linspace(0.15, 0.85, N)
            r = np.full(N, 0.05)
            break
    
    return np.column_stack((x, y, r))
