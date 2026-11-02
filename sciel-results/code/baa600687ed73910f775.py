"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Vectorized smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - L-BFGS-B optimization with analytical gradients
 - Multiple aspect ratios with joint refinement
 - Simulated annealing refinement
 - Final high-precision polish pass
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _softplus(x, beta=10.0):
    """Numerically stable softplus (scalar)."""
    if x > 1.0 / beta:
        return x
    return np.log1p(np.exp(beta * x)) / beta


def _softplus_vec(x, beta=10.0):
    """Vectorized numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 1.0 / beta
    out[mask] = x[mask]
    xm = x[~mask]
    out[~mask] = np.log1p(np.exp(beta * xm)) / beta
    return out


def _softplus_grad_vec(x, beta=10.0):
    """Vectorized derivative of softplus: sigmoid(beta*x)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * x))


def _compute_overlap_penalty(xs, ys, rs, beta=10.0):
    """Smooth overlap penalty using softplus surrogate (vectorized)."""
    # Compute pairwise distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    val = min_dist - dist
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    vals = val[iu]
    penalty = np.sum(_softplus_vec(vals, beta))
    return penalty


def _compute_overlap_penalty_grad(xs, ys, rs, beta=10.0):
    """Gradient of overlap penalty w.r.t. xs, ys, rs."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    val = min_dist - dist
    
    # sigmoid(beta * val)
    sig = _softplus_grad_vec(val, beta)
    
    # Gradient w.r.t. xs[i]: sum over j of sig * (xs[i]-xs[j])/dist
    # d(val_ij)/d(xs[i]) = -1 (since val = min_dist - dist, d(dist)/d(xs[i]) = dx/dist)
    # d(val_ij)/d(xs[i]) = -dx_ij / dist_ij
    # d(softplus)/d(xs[i]) = sig * d(val)/d(xs[i])
    
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # For each pair (i,j), i<j:
    # val_ij = rs[i]+rs[j] - dist_ij
    # d(softplus(val_ij))/d(xs[i]) = sig_ij * (-dx_ij / dist_ij)
    # d(softplus(val_ij))/d(xs[j]) = sig_ij * (dx_ij / dist_ij)
    # d(softplus(val_ij))/d(rs[i]) = sig_ij * 1
    # d(softplus(val_ij))/d(rs[j]) = sig_ij * 1
    
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    dx_ij = dx[i_idx, j_idx]
    dy_ij = dy[i_idx, j_idx]
    dist_ij = dist[i_idx, j_idx]
    sig_ij = sig[i_idx, j_idx]
    
    # Avoid division by zero
    inv_dist = np.where(dist_ij > 1e-10, 1.0 / dist_ij, 0.0)
    
    # Contribution to gx[i] from pair (i,j): -sig * dx_ij / dist_ij
    np.add.at(gx, i_idx, -sig_ij * dx_ij * inv_dist)
    np.add.at(gx, j_idx, sig_ij * dx_ij * inv_dist)
    
    np.add.at(gy, i_idx, -sig_ij * dy_ij * inv_dist)
    np.add.at(gy, j_idx, sig_ij * dy_ij * inv_dist)
    
    np.add.at(gr, i_idx, sig_ij)
    np.add.at(gr, j_idx, sig_ij)
    
    return gx, gy, gr


def _boundary_penalty(xs, ys, rs, w, h):
    """Squared boundary violation penalty (vectorized)."""
    pen = 0.0
    # Left boundary
    viol_l = np.maximum(rs - xs, 0.0)
    pen += np.sum(viol_l ** 2)
    # Right boundary
    viol_r = np.maximum(xs + rs - w, 0.0)
    pen += np.sum(viol_r ** 2)
    # Bottom boundary
    viol_b = np.maximum(rs - ys, 0.0)
    pen += np.sum(viol_b ** 2)
    # Top boundary
    viol_t = np.maximum(ys + rs - h, 0.0)
    pen += np.sum(viol_t ** 2)
    return pen


def _boundary_penalty_grad(xs, ys, rs, w, h):
    """Gradient of boundary penalty w.r.t. xs, ys, rs."""
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # Left: viol = max(rs - xs, 0), pen = viol^2
    # d(pen)/d(xs) = -2*viol when viol > 0
    # d(pen)/d(rs) = 2*viol when viol > 0
    viol_l = np.maximum(rs - xs, 0.0)
    mask_l = viol_l > 0
    gx[mask_l] -= 2.0 * viol_l[mask_l]
    gr[mask_l] += 2.0 * viol_l[mask_l]
    
    # Right: viol = max(xs + rs - w, 0)
    viol_r = np.maximum(xs + rs - w, 0.0)
    mask_r = viol_r > 0
    gx[mask_r] += 2.0 * viol_r[mask_r]
    gr[mask_r] += 2.0 * viol_r[mask_r]
    
    # Bottom: viol = max(rs - ys, 0)
    viol_b = np.maximum(rs - ys, 0.0)
    mask_b = viol_b > 0
    gy[mask_b] -= 2.0 * viol_b[mask_b]
    gr[mask_b] += 2.0 * viol_b[mask_b]
    
    # Top: viol = max(ys + rs - h, 0)
    viol_t = np.maximum(ys + rs - h, 0.0)
    mask_t = viol_t > 0
    gy[mask_t] += 2.0 * viol_t[mask_t]
    gr[mask_t] += 2.0 * viol_t[mask_t]
    
    return gx, gy, gr


def _objective(params, w, h, beta=10.0):
    """
    Objective: minimize penalty - sum(r).
    params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs <= 0):
        return 1e10 + np.sum(np.minimum(rs, 0) ** 2)
    
    overlap_pen = _compute_overlap_penalty(xs, ys, rs, beta)
    boundary_pen = _boundary_penalty(xs, ys, rs, w, h)
    
    return overlap_pen + boundary_pen - np.sum(rs)


def _objective_grad(params, w, h, beta=10.0):
    """Gradient of objective."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs <= 0):
        grad = np.zeros_like(params)
        grad[2::3] = 2.0 * np.minimum(rs, 0)
        return grad
    
    gx_o, gy_o, gr_o = _compute_overlap_penalty_grad(xs, ys, rs, beta)
    gx_b, gy_b, gr_b = _boundary_penalty_grad(xs, ys, rs, w, h)
    
    gx = gx_o + gx_b
    gy = gy_o + gy_b
    gr = gr_o + gr_b - 1.0  # derivative of -sum(rs)
    
    grad = np.zeros_like(params)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    return grad


def _validity_margin(result, tol=1e-4):
    """Return minimum clearance (positive = valid). result is (N,3): x, y, r."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    
    if bw + bh > 2.0 + tol:
        return -(bw + bh - 2.0)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            m = d - (rs[i] + rs[j])
            if m < -tol:
                return m
    
    return min(bw + bh - 2.0, 0.0)


def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    return result


def _check_and_update(params, w, h, best_state):
    """Check validity and update best if improved."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs < 1e-9):
        return
    if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
        return
    if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
        return
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < (min_dist - 1e-7) ** 2:
                return
    
    val = np.sum(rs)
    if val > best_state['value']:
        best_state['value'] = val
        best_state['result'] = np.zeros((N, 3))
        best_state['result'][:, 0] = xs
        best_state['result'][:, 1] = ys
        best_state['result'][:, 2] = rs


def _sa_optimize(w, h, x0, max_iters=2000, seed=42, deadline=None):
    """Simulated annealing for fixed w, h (vectorized cost eval)."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    r_max = min(w, h) / 2.0
    T = 0.05
    cooling = 0.999
    current_cost = _compute_overlap_penalty(xs, ys, rs, beta=10.0) - np.sum(rs)
    
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
        
        scale = T * 0.3
        xs[idx] += rng.normal(0, scale)
        ys[idx] += rng.normal(0, scale)
        rs[idx] += rng.normal(0, scale * 0.5)
        
        xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
        ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
        rs[idx] = np.clip(rs[idx], 1e-4, r_max)
        
        new_cost = _compute_overlap_penalty(xs, ys, rs, beta=10.0) - np.sum(rs)
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 1e-6 and rng.random() < np.exp(-delta / T)):
            current_cost = new_cost
        else:
            xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
        
        T *= cooling
    
    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def _grid_initial(w, h):
    """Create grid arrangement of 21 circles."""
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        col = i % a
        row = i // a
        x0[3 * i] = (col + 0.5) * w / a
        x0[3 * i + 1] = (row + 0.5) * h / b
        x0[3 * i + 2] = min(w / a, h / b) * 0.45
    
    return x0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Try multiple aspect ratios
    widths = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    
    best_x_for_warm = None
    
    for w in widths:
        if time.time() > deadline - 20.0:
            break
        h = 2.0 - w
        
        # Initialize with grid
        x0 = _grid_initial(w, h)
        
        # L-BFGS-B optimization with smooth objective and gradients
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        
        try:
            # First pass with moderate beta and gradient
            res = minimize(lambda p: _objective(p, w, h, beta=5.0), x0,
                           jac=lambda p: _objective_grad(p, w, h, beta=5.0),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
            if time.time() > deadline - 10.0:
                break
            _check_and_update(res.x, w, h, best_state)
            
            # Second pass with higher beta for more precise optimization
            if time.time() < deadline - 15.0:
                res2 = minimize(lambda p: _objective(p, w, h, beta=20.0), res.x,
                                jac=lambda p: _objective_grad(p, w, h, beta=20.0),
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12})
                _check_and_update(res2.x, w, h, best_state)
                if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                    best_x_for_warm = res2.x.copy()
        except Exception:
            pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 25.0:
            x_sa_start = best_x_for_warm if best_x_for_warm is not None else x0
            x_sa = _sa_optimize(w, h, x_sa_start, max_iters=2000, seed=42, deadline=deadline - 10.0)
            _check_and_update(x_sa, w, h, best_state)
            
            # Second SA pass with different seed
            if time.time() < deadline - 15.0:
                rng_sa = np.random.default_rng(123)
                x_sa2 = x_sa.copy()
                for i in range(N):
                    x_sa2[3 * i] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 1] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 2] += rng_sa.normal(0, 0.005)
                x_sa2 = _sa_optimize(w, h, x_sa2, max_iters=1500, seed=123, deadline=deadline - 10.0)
                _check_and_update(x_sa2, w, h, best_state)

    # Final high-precision polish pass
    if time.time() < deadline - 5.0 and best_state['result'] is not None:
        # Extract best parameters
        xs = best_state['result'][:, 0]
        ys = best_state['result'][:, 1]
        rs = best_state['result'][:, 2]
        
        # Determine w, h from bounding box
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y
        w_polish = bw
        h_polish = bh
        
        if w_polish > 0 and h_polish > 0:
            x_polish = np.zeros(3 * N)
            x_polish[0::3] = xs - min_x
            x_polish[1::3] = ys - min_y
            x_polish[2::3] = rs
            
            bounds_polish = []
            r_max_polish = min(w_polish, h_polish) / 2.0
            for i in range(N):
                bounds_polish.append((0, w_polish))
                bounds_polish.append((0, h_polish))
                bounds_polish.append((1e-4, r_max_polish))
            
            try:
                # High-precision polish with gradient
                res_polish = minimize(lambda p: _objective(p, w_polish, h_polish, beta=50.0), x_polish,
                                      jac=lambda p: _objective_grad(p, w_polish, h_polish, beta=50.0),
                                      method='L-BFGS-B', bounds=bounds_polish,
                                      options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14})
                _check_and_update(res_polish.x, w_polish, h_polish, best_state)
                
                # Even tighter polish
                if time.time() < deadline - 3.0:
                    res_polish2 = minimize(lambda p: _objective(p, w_polish, h_polish, beta=100.0), res_polish.x,
                                           jac=lambda p: _objective_grad(p, w_polish, h_polish, beta=100.0),
                                           method='L-BFGS-B', bounds=bounds_polish,
                                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
                    _check_and_update(res_polish2.x, w_polish, h_polish, best_state)
            except Exception:
                pass
    
    # Fallback if no valid result found
    if best_state['result'] is None or best_state['value'] <= 0:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r
        _scale_to_perimeter(best_result)
        return best_result
    
    # Scale to ensure perimeter <= 4
    result = best_state['result'].copy()
    _scale_to_perimeter(result)
    return result
