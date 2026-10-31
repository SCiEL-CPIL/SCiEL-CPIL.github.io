"""Circle packing in perimeter-4 rectangle: SLSQP + LSE Refine + Coordinate Descent."""
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


def _lse_refine(x, y, r, w, h, time_budget):
    """
    Log-sum-exp based refinement: jointly optimize positions and radii
    using a smooth surrogate for the nonsmooth packing constraints.
    """
    n = N
    # Parameters: x (n), y (n), r (n), w (1)
    z = np.concatenate([x, y, r, [w]])
    
    def objective(z):
        x_ = z[:n]
        y_ = z[n:2*n]
        r_ = z[2*n:3*n]
        w_ = z[3*n]
        h_ = 2.0 - w_
        
        if w_ < 0.2 or h_ < 0.2 or np.any(r_ < 0.001):
            return 1e6
        
        obj = -np.sum(r_)
        
        # Pairwise overlap penalty (LSE softplus)
        dx = x_[:, None] - x_[None, :]
        dy = y_[:, None] - y_[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_[:, None] + r_[None, :]
        
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = r_sum_pairs - d_pairs
        
        temp = 0.01
        soft_overlap = temp * np.log1p(np.exp(overlap / temp))
        obj += 500.0 * np.sum(soft_overlap)
        
        # Boundary penalties
        left_viol = r_ - x_
        right_viol = x_ + r_ - w_
        bottom_viol = r_ - y_
        top_viol = y_ + r_ - h_
        
        soft_left = temp * np.log1p(np.exp(left_viol / temp))
        soft_right = temp * np.log1p(np.exp(right_viol / temp))
        soft_bottom = temp * np.log1p(np.exp(bottom_viol / temp))
        soft_top = temp * np.log1p(np.exp(top_viol / temp))
        
        obj += 500.0 * (np.sum(soft_left) + np.sum(soft_right) + np.sum(soft_bottom) + np.sum(soft_top))
        
        # Keep aspect ratio in reasonable range
        if w_ < 0.5:
            obj += 100.0 * (0.5 - w_) ** 2
        elif w_ > 1.5:
            obj += 100.0 * (w_ - 1.5) ** 2
        
        return obj
    
    bounds = []
    for i in range(n):
        bounds.extend([(0.01, 2.0), (0.01, 2.0), (0.001, 0.5)])
    bounds.append((0.3, 1.7))
    
    maxiter = int(min(500, time_budget * 50))
    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        
        z_opt = res.x
        x_new = z_opt[:n]
        y_new = z_opt[n:2*n]
        r_new = z_opt[2*n:3*n]
        w_new = z_opt[3*n]
        h_new = 2.0 - w_new
        
        if _is_valid(x_new, y_new, r_new, w_new, h_new):
            return x_new, y_new, r_new, w_new
        else:
            # Try shrinking radii slightly
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r_new * factor
                if _is_valid(x_new, y_new, r_trial, w_new, h_new):
                    return x_new, y_new, r_trial, w_new
            return None
    except Exception:
        return None


def _coordinate_descent_polish(x, y, r, w, h, time_budget):
    """
    Coordinate descent: iteratively adjust one circle at a time to maximize
    the minimum gap, then grow radii.
    """
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    def min_gap(i):
        """Minimum gap for circle i to walls and other circles."""
        gap = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = min(gap, d - r[i] - r[j])
        return gap
    
    for _ in range(50):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            
            # Try moving circle i in 8 directions
            best_pos = (x[i], y[i])
            best_gap = min_gap(i)
            
            for dx_d in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                for dy_d in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                    if dx_d == 0 and dy_d == 0:
                        continue
                    new_x = x[i] + dx_d
                    new_y = y[i] + dy_d
                    if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                        continue
                    
                    # Check if move is valid (no overlaps)
                    valid_move = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d2 = dx * dx + dy * dy
                        r_sum = r[i] + r[j]
                        if d2 < r_sum * r_sum - 1e-9:
                            valid_move = False
                            break
                    
                    if not valid_move:
                        continue
                    
                    # Compute new min gap
                    gap = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        gap = min(gap, d - r[i] - r[j])
                    
                    if gap > best_gap + 1e-9:
                        best_gap = gap
                        best_pos = (new_x, new_y)
            
            if best_pos != (x[i], y[i]):
                x[i], y[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    # After position optimization, try to grow radii
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
        else:
            break
    
    if _is_valid(x, y, r, w, h):
        return x, y, r, w
    return None


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
        side = int(np.ceil(np.sqrt(N)))
        
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
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
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 30)))
        
        # SLSQP with LSE objective
        x = z0[0::3][:N]
        y = z0[1::3][:N]
        r = z0[2::3][:N]
        w = z0[-1]
        h = 2.0 - w
        
        # First, do a quick LSE refine
        lse_result = _lse_refine(x, y, r, w, h, remaining - 10)
        if lse_result is not None:
            x_new, y_new, r_new, w_new = lse_result
            valid, val = check_validity(np.concatenate([x_new, y_new, r_new, [w_new]]))
            if valid and val > best_value:
                best_value = val
                best_result = np.concatenate([x_new, y_new, r_new, [w_new]]).copy()
        
        # Then coordinate descent polish
        if time.time() < deadline - 15:
            cd_result = _coordinate_descent_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 10)
            if cd_result is not None:
                x_new, y_new, r_new, w_new = cd_result
                valid, val = check_validity(np.concatenate([x_new, y_new, r_new, [w_new]]))
                if valid and val > best_value:
                    best_value = val
                    best_result = np.concatenate([x_new, y_new, r_new, [w_new]]).copy()
    
    # Final polish on best result
    if best_result is not None and time.time() < deadline - 5:
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
        w = best_result[-1]
        h = 2.0 - w
        
        # LSE refine
        lse_result = _lse_refine(x, y, r, w, h, deadline - (time.time() - start_time) - 5)
        if lse_result is not None:
            x_new, y_new, r_new, w_new = lse_result
            valid, val = check_validity(np.concatenate([x_new, y_new, r_new, [w_new]]))
            if valid and val > best_value:
                best_value = val
                best_result = np.concatenate([x_new, y_new, r_new, [w_new]]).copy()
        
        # Coordinate descent
        if time.time() < deadline - 2:
            cd_result = _coordinate_descent_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 1)
            if cd_result is not None:
                x_new, y_new, r_new, w_new = cd_result
                valid, val = check_validity(np.concatenate([x_new, y_new, r_new, [w_new]]))
                if valid and val > best_value:
                    best_value = val
                    best_result = np.concatenate([x_new, y_new, r_new, [w_new]]).copy()
    
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
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
