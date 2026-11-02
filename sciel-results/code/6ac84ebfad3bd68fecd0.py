"""Circle packing in perimeter-4 rectangle: Rigid-body simulation + SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _simulation_init(w, h, n_circles, seed=42):
    """
    Rigid-body simulation to find a dense packing.
    Uses a symplectic-like integrator with collision resolution.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize circles randomly within the rectangle
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)
    
    # Start with small circles and grow them
    for i in range(n_circles):
        r[i] = 0.02
        x[i] = rng.uniform(r[i], w - r[i])
        y[i] = rng.uniform(r[i], h - r[i])
    
    # Simulation parameters
    dt = 0.01
    num_steps = 2000
    growth_rate = 0.001
    damping = 0.99
    corner_bias = 0.5
    
    for step in range(num_steps):
        # Grow radii slowly
        for i in range(n_circles):
            r[i] += growth_rate
        
        # Compute forces
        fx = np.zeros(n_circles)
        fy = np.zeros(n_circles)
        
        # Pairwise repulsion (Hooke's law for overlapping circles)
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                
                if d < min_d and d > 1e-10:
                    # Repulsive force proportional to overlap
                    force = 100.0 * (min_d - d)
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d
        
        # Boundary forces (push away from walls)
        for i in range(n_circles):
            # Left wall
            if x[i] < r[i]:
                fx[i] += 50.0 * (r[i] - x[i])
            # Right wall
            if x[i] > w - r[i]:
                fx[i] -= 50.0 * (x[i] - (w - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += 50.0 * (r[i] - y[i])
            # Top wall
            if y[i] > h - r[i]:
                fy[i] -= 50.0 * (y[i] - (h - r[i]))
        
        # Corner bias: attract circles toward corners
        for i in range(n_circles):
            # Distance to each corner
            corners = [(0, 0), (w, 0), (0, h), (w, h)]
            min_corner_dist = float('inf')
            nearest_corner = None
            
            for cx, cy in corners:
                dist = np.sqrt((x[i] - cx) ** 2 + (y[i] - cy) ** 2)
                if dist < min_corner_dist:
                    min_corner_dist = dist
                    nearest_corner = (cx, cy)
            
            # Bias force toward nearest corner (weak)
            bias_strength = 0.01
            fx[i] += bias_strength * (nearest_corner[0] - x[i])
            fy[i] += bias_strength * (nearest_corner[1] - y[i])
        
        # Integrate (symplectic Euler)
        for i in range(n_circles):
            x[i] += fx[i] * dt
            y[i] += fy[i] * dt
        
        # Damping
        # (In a real simulation we'd have velocity, but here we're using position-based dynamics)
        # Clamp to boundaries
        for i in range(n_circles):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    return x, y, r


def _force_relax(x, y, r, w, h, max_iter=1000, max_time=10.0, start_time=None):
    """Force-directed relaxation to resolve overlaps."""
    if start_time is None:
        start_time = time.time()
    
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    n = len(r)
    dt = 0.01
    
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-10
                min_d = r[i] + r[j]
                if d < min_d:
                    force = (min_d - d) / d
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        fx -= 10.0 * left_viol
        fx += 10.0 * right_viol
        fy -= 10.0 * bottom_viol
        fy += 10.0 * top_viol
        
        # Update
        x += dt * fx
        y += dt * fy
        
        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)))
        if max_force < 1e-10:
            break
    
    return x, y, r


def _expand_radii(x, y, r, w, h, max_time=20.0, start_time=None):
    """Try to expand each circle's radius while maintaining validity."""
    if start_time is None:
        start_time = time.time()
    
    n = len(r)
    step = 0.001
    
    for it in range(500):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(n):
            new_r = r.copy()
            new_r[i] += step
            
            if new_r[i] > w / 2 or new_r[i] > h / 2:
                continue
            
            # Check if expanding is valid
            valid = True
            if new_r[i] > x[i] or x[i] + new_r[i] > w:
                valid = False
            elif new_r[i] > y[i] or y[i] + new_r[i] > h:
                valid = False
            else:
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < new_r[i] + r[j] - 1e-9:
                        valid = False
                        break
            
            if valid:
                r[i] = new_r[i]
                improved = True
                break
        
        if not improved:
            step *= 0.7
            if step < 1e-7:
                break
    
    return r


def _slsqp_optimize(x, y, r, w, h, max_time=30.0, start_time=None):
    """SLSQP optimization to maximize sum of radii."""
    if start_time is None:
        start_time = time.time()
    
    n = len(r)
    
    # Pack into vector: [x0, y0, r0, x1, y1, r1, ..., xn, yn, rn, w]
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w
    
    def objective(z_):
        x_ = z_[0::3][:n]
        y_ = z_[1::3][:n]
        r_ = z_[2::3][:n]
        w_ = z_[-1]
        h_ = 2.0 - w_
        
        if w_ < 0.2 or h_ < 0.2 or np.any(r_ < 0):
            return 1e6
        
        obj = -np.sum(r_)
        
        # Overlap penalty
        penalty_weight = 1000.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_[i] - x_[j]
                dy = y_[i] - y_[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = max(0, r_[i] + r_[j] - d)
                obj += penalty_weight * overlap * overlap
        
        # Boundary penalty
        for i in range(n):
            left_viol = max(0, r_[i] - x_[i])
            right_viol = max(0, x_[i] + r_[i] - w_)
            bottom_viol = max(0, r_[i] - y_[i])
            top_viol = max(0, y_[i] + r_[i] - h_)
            
            obj += penalty_weight * (left_viol ** 2 + right_viol ** 2 + bottom_viol ** 2 + top_viol ** 2)
        
        return obj
    
    bounds = [(0, 2.0)] * (3 * n) + [(0.3, 1.7)]
    
    try:
        remaining = max_time - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 30)))
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        return res.x
    except Exception:
        return z


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_result = None
    
    def check_validity(x, y, r, w, h, tol=1e-9):
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
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Phase 1: Simulation initialization
        x, y, r = _simulation_init(w0, h0, N, seed=int(w0 * 100))
        
        # Phase 2: Force relaxation to resolve any remaining overlaps
        x, y, r = _force_relax(x, y, r, w0, h0, max_iter=500, max_time=5.0)
        
        # Phase 3: Expand radii
        r = _expand_radii(x, y, r, w0, h0, max_time=10.0)
        
        # Check validity after expansion
        valid, val = check_validity(x, y, r, w0, h0)
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack((x, y, r))
        
        # Phase 4: SLSQP optimization
        if time.time() < deadline - 15:
            z_opt = _slsqp_optimize(x, y, r, w0, h0, max_time=15.0)
            
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt
            
            valid, val = check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack((x_opt, y_opt, r_opt))
            elif not valid:
                # Try repair: shrink radii slightly
                r_repair = r_opt * 0.95
                valid, val = check_validity(x_opt, y_opt, r_repair, w_opt, h_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = np.column_stack((x_opt, y_opt, r_repair))
    
    # Phase 5: Final polish on best result
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        
        # Try to find the best w that fits these circles
        # The bounding box should have perimeter <= 4
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        w_box = max_x - min_x
        h_box = max_y - min_y
        perim = 2 * (w_box + h_box)
        
        if perim > 4.0:
            # Scale down
            scale = 2.0 / (w_box + h_box)
            if scale < 1.0:
                # Shrink radii
                r *= scale
                x = (x - (min_x + max_x) / 2) * scale + 1.0
                y = (y - (min_y + max_y) / 2) * scale + 1.0
        
        valid, val = check_validity(x, y, r, 1.0, 1.0)
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack((x, y, r))
    
    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final safety: ensure strict validity with minimal shrink
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, 1.0, 1.0):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
