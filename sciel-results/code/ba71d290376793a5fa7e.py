import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1+exp(beta*x))/beta"""
    xb = beta * x
    return np.where(xb > 0, 
                    x + np.log1p(np.exp(-xb)) / beta, 
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)"""
    xb = beta * x
    return np.where(xb > 0, 
                    1.0 / (1.0 + np.exp(-xb)), 
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_and_grad(x, w, h, beta=100.0, var_weight=0.0):
    """
    Objective: -sum(r) + penalties for violations.
    x: [x1, y1, r1, x2, y2, r2, ...]
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    
    # 1. Boundary Constraints
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # 2. Pairwise Non-overlap Constraints
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-12)
    
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    v = r_sum - d
    
    iu = np.triu_indices(N, k=1)
    v_upper = v[iu]
    d_upper = d[iu]
    dx_upper = dx[iu]
    dy_upper = dy[iu]
    
    pen = _softplus(v_upper, beta)
    obj += np.sum(pen)
    
    g = _softplus_grad(v_upper, beta)
    
    idx_i = iu[0]
    idx_j = iu[1]
    
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        gk = g[k]
        dk = d_upper[k]
        dxk = dx_upper[k]
        dyk = dy_upper[k]
        
        grad[3*i]   += -gk * dxk / dk
        grad[3*i+1] += -gk * dyk / dk
        grad[3*i+2] +=  gk
        
        grad[3*j]   +=  gk * dxk / dk
        grad[3*j+1] +=  gk * dyk / dk
        grad[3*j+2] +=  gk

    # 3. Variance Penalty (Equalization Pressure)
    if var_weight > 0.0:
        r_mean = np.mean(rs)
        diff = rs - r_mean
        obj += var_weight * np.sum(diff * diff)
        grad[2::3] += 2.0 * var_weight * diff

    # 4. Objective gradient for -sum(r)
    grad[2::3] += -1.0

    return obj, grad

def _strictly_valid(circles, w, h, tol=1e-8):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d2 = dx * dx + dy * dy
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    iu = np.triu_indices(n, k=1)
    d2_upper = d2[iu]
    r_sum_upper = r_sum[iu]
    if np.any(d2_upper < (r_sum_upper - tol) ** 2):
        return False
    return True

def _hex_init(w, h, seed):
    """Generate hex-packed initial configurations."""
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3],
        [6, 5, 5, 5],
        [5, 6, 5, 5],
        [5, 5, 6, 5],
        [5, 5, 5, 6],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        est_r_h = w / (max_cols * 2.0 + 0.5)
        est_r_v = h / ((n_rows - 1) * np.sqrt(3) + 2.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
            
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = r_est + row_idx * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best

def _grid_init(w, h, seed):
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)

def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles

def _physics_refine(circles, w, h, deadline, iterations=500, k_rep=10.0, k_attr=5.0, damping=0.9):
    """
    Physics-based force-directed refinement.
    Uses repulsive forces for overlapping circles and attractive forces for nearby non-overlapping circles
    to help escape local minima and find denser packings.
    """
    current = circles.copy()
    n = current.shape[0]
    
    for it in range(iterations):
        if time.time() > deadline:
            break
            
        forces = np.zeros_like(current)
        
        # Repulsive forces (for overlapping or close circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = current[i, 0] - current[j, 0]
                dy = current[i, 1] - current[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-6:
                    d = 1e-6
                    dx = 1e-6
                    dy = 0.0
                r_sum = current[i, 2] + current[j, 2]
                
                if d < r_sum * 1.5:  # Only interact if reasonably close
                    # Strong repulsion if overlapping, weak repulsion if close
                    if d < r_sum:
                        f = k_rep * (r_sum - d)
                    else:
                        f = -k_attr * (d - r_sum) * 0.1  # Weak attraction to pack tightly
                    
                    fx = f * dx / d
                    fy = f * dy / d
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall constraints: push back if too close to walls
        for i in range(n):
            x, y, r = current[i]
            if x - r < 0.05:
                forces[i, 0] += k_rep * (0.05 - (x - r))
            if x + r > w - 0.05:
                forces[i, 0] -= k_rep * ((x + r) - (w - 0.05))
            if y - r < 0.05:
                forces[i, 1] += k_rep * (0.05 - (y - r))
            if y + r > h - 0.05:
                forces[i, 1] -= k_rep * ((y + r) - (h - 0.05))
        
        # Update positions with damping
        step_size = 0.01
        current[:, 0] += step_size * forces[:, 0] * damping
        current[:, 1] += step_size * forces[:, 1] * damping
        
        # Clamp to valid bounds
        for i in range(n):
            current[i, 0] = np.clip(current[i, 0], current[i, 2], w - current[i, 2])
            current[i, 1] = np.clip(current[i, 1], current[i, 2], h - current[i, 2])
    
    return current

def _scheduled_optimize(x0, w, h, deadline, var_weight=0.0, use_physics=True):
    """Run optimization with increasing beta, optionally with physics-based refinement."""
    # Phases: (beta, max_iter, var_weight)
    phases = [
        (10.0, 50, 0.01),
        (50.0, 50, 0.01),
        (200.0, 50, 0.001),
        (500.0, 100, 0.0),
        (1000.0, 100, 0.0)
    ]
    
    current_x = x0
    for beta, max_iter, vw in phases:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(w, h, beta, vw),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            current_x = res.x
        except Exception:
            pass
        
        # Optional physics-based refinement after each phase
        if use_physics and time.time() < deadline - 5:
            c = current_x.reshape(-1, 3)
            c_refined = _physics_refine(c, w, h, deadline, iterations=200)
            if _strictly_valid(c_refined, w, h):
                current_x = c_refined.flatten()
    
    return current_x

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None

    # Aspect ratios to try
    aspect_ratios = [
        1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5,
        0.85, 1.15, 0.75, 1.25, 0.65, 1.35, 0.55, 1.45, 0.95, 1.05,
        0.92, 1.08, 0.88, 1.12, 0.82, 1.18, 0.78, 1.22
    ]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break

        a = 1.0 / ar  # h/w
        w = 2.0 / (1.0 + a)
        h = 2.0 * a / (1.0 + a)

        # Generate initializations
        inits = []
        inits.append(_hex_init(w, h, 0))
        inits.append(_grid_init(w, h, 0))
        inits.append(_random_init(w, h, 42))

        for init in inits:
            if time.time() > deadline:
                break
            
            x0 = init.flatten()
            
            # Scheduled optimization with physics refinement
            x_opt = _scheduled_optimize(x0, w, h, deadline)
            
            c = x_opt.reshape(-1, 3)
            c[:, 0] = np.clip(c[:, 0], 0, w)
            c[:, 1] = np.clip(c[:, 1], 0, h)
            c[:, 2] = np.clip(c[:, 2], 1e-9, min(w, h)/2)
            
            if _strictly_valid(c, w, h):
                val = float(np.sum(c[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = c.copy()
                    best_w, best_h = w, h

    if best_result is None:
        w, h = 1.0, 1.0
        a = 5
        b = 5
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = 0.15
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best_result = np.array(circles)
        best_val = float(np.sum(best_result[:, 2]))
        best_w, best_h = w, h

    # Final polish: Run one more high-beta optimization on the best found configuration
    if time.time() < deadline - 5:
        x_best = best_result.flatten()
        x_polished = _scheduled_optimize(x_best, best_w, best_h, deadline, var_weight=0.0, use_physics=True)
        c_polished = x_polished.reshape(-1, 3)
        c_polished[:, 0] = np.clip(c_polished[:, 0], 0, best_w)
        c_polished[:, 1] = np.clip(c_polished[:, 1], 0, best_h)
        c_polished[:, 2] = np.clip(c_polished[:, 2], 1e-9, min(best_w, best_h)/2)
        
        if _strictly_valid(c_polished, best_w, best_h):
            val_polished = float(np.sum(c_polished[:, 2]))
            if val_polished > best_val:
                best_val = val_polished
                best_result = c_polished.copy()

    return best_result
