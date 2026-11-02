"""Softplus-penalty + L-BFGS-B circle packing solver for perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0 / beta, x, np.logaddexp(0, beta * x) / beta)

def _pack_objective(params, beta=50.0):
    """
    Objective: minimize -sum(r_i) + penalty for constraint violations.
    params: [w, x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
    w = width, h = 2 - w
    """
    w = params[0]
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return 1e10
    xs = params[1:1+N]
    ys = params[1+N:1+2*N]
    rs = params[1+2*N:1+3*N]
    
    # Sum of radii (we maximize, so minimize negative)
    score = -np.sum(rs)
    
    # Penalty for wall violations
    # Left: x_i >= r_i  => violation = r_i - x_i
    # Right: x_i + r_i <= w => violation = x_i + r_i - w
    # Bottom: y_i >= r_i
    # Top: y_i + r_i <= h
    pen = 0.0
    pen += np.sum(_softplus(rs - xs, beta))
    pen += np.sum(_softplus(xs + rs - w, beta))
    pen += np.sum(_softplus(rs - ys, beta))
    pen += np.sum(_softplus(ys + rs - h, beta))
    
    # Penalty for circle-circle overlaps
    # For each pair (i,j): dist_ij >= r_i + r_j
    # violation = r_i + r_j - dist_ij
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-15)
            pen += _softplus(rs[i] + rs[j] - dist, beta)
    
    # Penalty for negative radii
    pen += np.sum(_softplus(-rs, beta))
    
    # Penalty for w out of range
    pen += _softplus(0.1 - w, beta) + _softplus(w - 1.9, beta)
    
    return score + pen

def _pack_objective_grad(params, beta=50.0):
    """Gradient of pack_objective, vectorized where possible."""
    w = params[0]
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        grad = np.zeros_like(params)
        grad[0] = 1e8
        return grad
    
    xs = params[1:1+N].copy()
    ys = params[1+N:1+2*N].copy()
    rs = params[1+2*N:1+3*N].copy()
    
    grad = np.zeros_like(params)
    
    # d/dw of score = 0 (score doesn't depend on w directly)
    # But penalties depend on w
    
    # Helper: d softplus/dx = sigmoid(beta * x)
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    # Wall penalties
    # Left: softplus(r_i - x_i)
    v_left = rs - xs
    s_left = _sigmoid(beta * v_left)
    grad[1:1+N] -= s_left  # d/dx_i
    grad[1+2*N:1+3*N] += s_left  # d/dr_i
    
    # Right: softplus(x_i + r_i - w)
    v_right = xs + rs - w
    s_right = _sigmoid(beta * v_right)
    grad[1:1+N] += s_right  # d/dx_i
    grad[1+2*N:1+3*N] += s_right  # d/dr_i
    grad[0] -= np.sum(s_right)  # d/dw
    
    # Bottom: softplus(r_i - y_i)
    v_bot = rs - ys
    s_bot = _sigmoid(beta * v_bot)
    grad[1+N:1+2*N] -= s_bot
    grad[1+2*N:1+3*N] += s_bot
    
    # Top: softplus(y_i + r_i - h) where h = 2 - w
    v_top = ys + rs - h
    s_top = _sigmoid(beta * v_top)
    grad[1+N:1+2*N] += s_top
    grad[1+2*N:1+3*N] += s_top
    # d/dw: h = 2-w, so d/dw of (y+r-h) = d/dw of (y+r-2+w) = +1
    grad[0] += np.sum(s_top)
    
    # Circle-circle overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-15)
            v = rs[i] + rs[j] - dist
            s = _sigmoid(beta * v)
            if s > 1e-10:
                # d/dx_i of v = -dx/dist
                # d/dx_j of v = +dx/dist
                # d/dr_i of v = 1
                # d/dr_j of v = 1
                inv_dist = 1.0 / dist
                gx_i = -dx * inv_dist
                gy_i = -dy * inv_dist
                gx_j = dx * inv_dist
                gy_j = dy * inv_dist
                
                grad[1+i] += s * gx_i
                grad[1+N+i] += s * gy_i
                grad[1+j] += s * gx_j
                grad[1+N+j] += s * gy_j
                grad[1+2*N+i] += s
                grad[1+2*N+j] += s
    
    # Negative radius penalty
    v_neg = -rs
    s_neg = _sigmoid(beta * v_neg)
    grad[1+2*N:1+3*N] -= s_neg
    
    # w range penalty
    v_w_min = 0.1 - w
    s_w_min = _sigmoid(beta * v_w_min)
    grad[0] -= s_w_min
    v_w_max = w - 1.9
    s_w_max = _sigmoid(beta * v_w_max)
    grad[0] += s_w_max
    
    return grad

def _make_hex(w, h):
    """Hexagonal packing initialization."""
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    # Try to fit hex pattern
    cols = max(1, int(np.ceil(np.sqrt(N * w / h))))
    rows = max(1, int(np.ceil(N / cols)))
    
    idx = 0
    for row in range(rows):
        offset = (row % 2) * 0.5
        for col in range(cols):
            if idx >= N:
                break
            x = (col + offset + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.35 * min(cell_w, cell_h)
            xs[idx] = x
            ys[idx] = y
            rs[idx] = r
            idx += 1
    
    # Fill remaining
    while idx < N:
        xs[idx] = w * 0.5
        ys[idx] = h * 0.5
        rs[idx] = 0.1
        idx += 1
    
    return xs, ys, rs

def _make_grid(w, h):
    """Grid packing initialization."""
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    cols = max(1, int(np.ceil(np.sqrt(N * w / h))))
    rows = max(1, int(np.ceil(N / cols)))
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= N:
                break
            xs[idx] = (col + 0.5) * w / cols
            ys[idx] = (row + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            rs[idx] = 0.4 * min(cell_w, cell_h)
            idx += 1
    
    while idx < N:
        xs[idx] = w * 0.5
        ys[idx] = h * 0.5
        rs[idx] = 0.1
        idx += 1
    
    return xs, ys, rs

def _make_random(w, h, rng):
    """Random initialization."""
    xs = rng.uniform(0.1, w - 0.1, N)
    ys = rng.uniform(0.1, h - 0.1, N)
    rs = rng.uniform(0.05, 0.2, N)
    return xs, ys, rs

def _check_validity(xs, ys, rs, w, h, tol=1e-8):
    """Check if packing is valid."""
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(rs < -tol):
        return False
    if np.any(xs < -tol) or np.any(xs > w + tol):
        return False
    if np.any(ys < -tol) or np.any(ys > h + tol):
        return False
    if np.any(xs < rs - tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys < rs - tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy < (rs[i] + rs[j])**2 - tol:
                return False
    return True

def _polish_gradient(xs, ys, rs, w, h, rng, time_budget, start_time, deadline):
    """High-precision polish using gradient descent with adaptive step size."""
    best_score = np.sum(rs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()
    best_w = w
    
    step_size = 0.01
    patience = 0
    max_patience = 200
    
    while time.time() < deadline - 2.0:
        if patience > max_patience:
            break
        
        # Compute gradient
        params = np.concatenate(([w], xs, ys, rs))
        grad = _pack_objective_grad(params, beta=200.0)
        
        # Clip gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100:
            grad = grad / grad_norm * 100
        
        # Take step
        new_params = params - step_size * grad
        new_w = new_params[0]
        new_xs = new_params[1:1+N]
        new_ys = new_params[1+N:1+2*N]
        new_rs = new_params[1+2*N:1+3*N]
        
        # Ensure positivity
        new_rs = np.maximum(new_rs, 0.01)
        new_w = np.clip(new_w, 0.1, 1.9)
        new_h = 2.0 - new_w
        new_xs = np.clip(new_xs, 0, new_w)
        new_ys = np.clip(new_ys, 0, new_h)
        
        # Check if improved
        new_obj = _pack_objective(new_params, beta=200.0)
        old_obj = _pack_objective(params, beta=200.0)
        
        if new_obj < old_obj:
            # Check validity for score tracking
            if _check_validity(new_xs, new_ys, new_rs, new_w, 2.0 - new_w):
                new_score = np.sum(new_rs)
                if new_score > best_score:
                    best_score = new_score
                    best_xs = new_xs.copy()
                    best_ys = new_ys.copy()
                    best_rs = new_rs.copy()
                    best_w = new_w
                    patience = 0
                else:
                    patience += 1
            else:
                patience += 1
            xs, ys, rs, w = new_xs, new_ys, new_rs, new_w
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.8
            patience += 1
    
    return best_xs, best_ys, best_rs, best_w, best_score

def _try_optimize(xs, ys, rs, w, rng, time_budget, start_time, deadline):
    """Run L-BFGS-B optimization starting from given state."""
    h = 2.0 - w
    params = np.concatenate(([w], xs, ys, rs))
    
    bounds = [(0.1, 1.9)] + [(0, 2.0)]*(2*N) + [(0.001, 1.0)]*N
    
    best_score = -np.inf
    best_params = params.copy()
    
    # Check initial validity
    if _check_validity(xs, ys, rs, w, h):
        best_score = np.sum(rs)
    
    # Phase 1: L-BFGS-B with moderate beta
    t_end = min(start_time + time_budget * 0.4, deadline - 30)
    while time.time() < t_end:
        try:
            result = minimize(
                _pack_objective, params,
                jac=_pack_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 40, 'maxfun': 5000}
            )
            params = result.x
            # Clip to valid range
            params[0] = np.clip(params[0], 0.1, 1.9)
            for i in range(N):
                params[1+i] = np.clip(params[1+i], 0, 2.0)
                params[1+N+i] = np.clip(params[1+N+i], 0, 2.0)
                params[1+2*N+i] = np.clip(params[1+2*N+i], 0.001, 1.0)
            
            w_c = params[0]
            xs_c = params[1:1+N]
            ys_c = params[1+N:1+2*N]
            rs_c = params[1+2*N:1+3*N]
            
            if _check_validity(xs_c, ys_c, rs_c, w_c, 2.0 - w_c):
                score_c = np.sum(rs_c)
                if score_c > best_score:
                    best_score = score_c
                    best_params = params.copy()
        except Exception:
            break
    
    # Phase 2: Higher precision L-BFGS-B
    t_end2 = min(start_time + time_budget * 0.7, deadline - 15)
    if time.time() < t_end2:
        try:
            result = minimize(
                lambda p: _pack_objective(p, beta=100.0), best_params,
                jac=lambda p: _pack_objective_grad(p, beta=100.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                         'maxls': 50, 'maxfun': 10000}
            )
            params = result.x
            params[0] = np.clip(params[0], 0.1, 1.9)
            for i in range(N):
                params[1+i] = np.clip(params[1+i], 0, 2.0)
                params[1+N+i] = np.clip(params[1+N+i], 0, 2.0)
                params[1+2*N+i] = np.clip(params[1+2*N+i], 0.001, 1.0)
            
            w_c = params[0]
            xs_c = params[1:1+N]
            ys_c = params[1+N:1+2*N]
            rs_c = params[1+2*N:1+3*N]
            
            if _check_validity(xs_c, ys_c, rs_c, w_c, 2.0 - w_c):
                score_c = np.sum(rs_c)
                if score_c > best_score:
                    best_score = score_c
                    best_params = params.copy()
        except Exception:
            pass
    
    # Phase 3: Gradient polish
    if time.time() < deadline - 5:
        w_p = best_params[0]
        xs_p = best_params[1:1+N]
        ys_p = best_params[1+N:1+2*N]
        rs_p = best_params[1+2*N:1+3*N]
        
        polish_time = min(time_budget * 0.3, deadline - time.time() - 3)
        if polish_time > 1.0:
            xs_f, ys_f, rs_f, w_f, score_f = _polish_gradient(
                xs_p, ys_p, rs_p, w_p, rng, polish_time, start_time, deadline
            )
            if score_f > best_score:
                best_score = score_f
                best_params = np.concatenate(([w_f], xs_f, ys_f, rs_f))
    
    return best_params, best_score

def construct():
    """Pack 21 circles with disjoint interiors in a rectangle of perimeter 4,
    maximizing the sum of their radii.
    Returns array of shape (n, 3): [x, y, r] for each circle.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_params = None
    
    rng = np.random.default_rng(12345)
    
    # Strategy: Try multiple initializations and aspect ratios
    # Phase A: Diverse initializations (quick)
    # Phase B: Refine best candidates (slower)
    # Phase C: Ultra-high-precision polish on best
    
    initializations = []
    
    # Hexagonal
    for w in [0.8, 1.0, 1.2, 1.4, 0.7, 1.3]:
        h = 2.0 - w
        xs, ys, rs = _make_hex(w, h)
        initializations.append((w, xs, ys, rs, 'hex'))
    
    # Grid
    for w in [0.8, 1.0, 1.2, 1.4, 0.7, 1.3]:
        h = 2.0 - w
        xs, ys, rs = _make_grid(w, h)
        initializations.append((w, xs, ys, rs, 'grid'))
    
    # Random
    for w in [0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
        h = 2.0 - w
        xs, ys, rs = _make_random(w, h, rng)
        initializations.append((w, xs, ys, rs, 'rand'))
    
    # Shuffle
    rng.shuffle(initializations)
    
    # Phase A: Quick optimization on diverse starts
    # Time budget per start: ~3s for quick scan
    time_per_start = 3.0
    results = []
    
    for w, xs, ys, rs, init_type in initializations:
        if time.time() > deadline - 40:  # Reserve 40s for refinement
            break
        t_start = time.time()
        try:
            params, score = _try_optimize(xs, ys, rs, w, rng, time_per_start, start_time, deadline)
            if score > best_score:
                best_score = score
                best_params = params.copy()
            if score > -1e9:
                results.append((score, w, xs, ys, rs, init_type))
        except Exception:
            continue
    
    # Sort results by score
    results.sort(key=lambda x: -x[0])
    
    # Phase B: Refine top candidates with more time
    if len(results) > 0:
        top_k = min(5, len(results))
        remaining_time = deadline - time.time() - 20  # Reserve 20s for final polish
        time_per_refine = max(5.0, remaining_time / top_k)
        
        for score, w, xs, ys, rs, init_type in results[:top_k]:
            if time.time() > deadline - 25:
                break
            try:
                params, new_score = _try_optimize(xs, ys, rs, w, rng, time_per_refine, start_time, deadline)
                if new_score > best_score:
                    best_score = new_score
                    best_params = params.copy()
            except Exception:
                continue
    
    # Phase C: Ultra-high-precision polish on best
    if best_params is not None and time.time() < deadline - 5:
        w_b = best_params[0]
        xs_b = best_params[1:1+N]
        ys_b = best_params[1+N:1+2*N]
        rs_b = best_params[1+2*N:1+3*N]
        
        polish_time = deadline - time.time() - 3
        if polish_time > 2.0:
            try:
                xs_f, ys_f, rs_f, w_f, score_f = _polish_gradient(
                    xs_b, ys_b, rs_b, w_b, rng, polish_time, start_time, deadline
                )
                if score_f > best_score:
                    best_score = score_f
                    best_params = np.concatenate(([w_f], xs_f, ys_f, rs_f))
            except Exception:
                pass
    
    # Fallback
    if best_params is None:
        # Simple grid fallback
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i+0.5)/side, (j+0.5)/side, 0.35/side])
        best_params = np.array([w] + [c[0] for c in cells] + [c[1] for c in cells] + [c[2] for c in cells])
        best_score = np.sum([c[2] for c in cells])
    
    # Extract final result
    w_final = best_params[0]
    xs_final = best_params[1:1+N]
    ys_final = best_params[1+N:1+2*N]
    rs_final = best_params[1+2*N:1+3*N]
    
    # Final validity check and repair
    h_final = 2.0 - w_final
    # Clip to bounds
    xs_final = np.clip(xs_final, 0, w_final)
    ys_final = np.clip(ys_final, 0, h_final)
    rs_final = np.maximum(rs_final, 0.001)
    
    # Ensure wall constraints
    for i in range(N):
        rs_final[i] = min(rs_final[i], xs_final[i], w_final - xs_final[i], ys_final[i], h_final - ys_final[i])
        rs_final[i] = max(rs_final[i], 0.001)
        xs_final[i] = np.clip(xs_final[i], rs_final[i], w_final - rs_final[i])
        ys_final[i] = np.clip(ys_final[i], rs_final[i], h_final - rs_final[i])
    
    result = np.column_stack((xs_final, ys_final, rs_final))
    
    return result
