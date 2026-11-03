import numpy as np
import time
import random
from scipy.optimize import minimize, linprog

N = 21
TIME_LIMIT = 145.0

def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0, 
                    x + np.log1p(np.exp(-xb)) / beta, 
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0, 
                    1.0 / (1.0 + np.exp(-xb)), 
                    np.exp(xb) / (1.0 + np.exp(xb)))

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

def _hex_init(w, h):
    """Generate hex-packed initial configuration for a given w, h."""
    # Estimate radius
    r_est = np.sqrt(0.9 * (w * h) / (N * np.pi))
    
    circles = []
    r = r_est
    for attempt in range(10):
        circles = []
        y = r
        row_idx = 0
        while y <= h - r + 1e-9:
            x_offset = 0.0 if row_idx % 2 == 0 else r
            x = r + x_offset
            while x <= w - r + 1e-9:
                circles.append([x, y, r])
                x += 2.0 * r
            y += np.sqrt(3) * r
            row_idx += 1
        
        if len(circles) >= N:
            break
        r *= 0.95
        
    if len(circles) < N:
        # Fallback grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
            
    circles = circles[:N]
    return np.array(circles)

def _objective_fixed_rect(x, w, h, beta=100.0):
    """
    Objective for fixed w, h.
    x: [x1, y1, r1, ...]
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    
    # Boundary
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

    # Pairwise
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

    # Gradient for -sum(r)
    grad[2::3] += -1.0

    return obj, grad

def _solve_lp_radii(circles, w, h):
    """
    Solve LP for optimal radii given fixed centers.
    Returns array of radii or None if infeasible.
    """
    xs = circles[:, 0]
    ys = circles[:, 1]
    
    c_obj = -np.ones(N)
    
    A_ub = []
    b_ub = []
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(xs[i])
        
        A_ub.append(row.copy())
        b_ub.append(w - xs[i])
        
        A_ub.append(row.copy())
        b_ub.append(ys[i])
        
        A_ub.append(row.copy())
        b_ub.append(h - ys[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _run_optimization_for_box(w, h, init_circles, deadline):
    """
    Run multi-phase L-BFGS for fixed box.
    """
    x_init = init_circles.flatten()
    current_x = x_init
    
    phases = [
        (10.0, 50),
        (50.0, 50),
        (200.0, 50),
        (500.0, 50),
        (1000.0, 50)
    ]
    
    best_val = -1e10
    best_circles = None
    
    for beta, max_iter in phases:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _objective_fixed_rect,
                current_x,
                args=(w, h, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            current_x = res.x
        except Exception:
            pass
        
        c_curr = current_x.reshape(-1, 3)
        c_curr[:, 0] = np.clip(c_curr[:, 0], 0, w)
        c_curr[:, 1] = np.clip(c_curr[:, 1], 0, h)
        c_curr[:, 2] = np.clip(c_curr[:, 2], 1e-9, min(w, h)/2)
        
        if _strictly_valid(c_curr, w, h):
            val = float(np.sum(c_curr[:, 2]))
            if val > best_val:
                best_val = val
                best_circles = c_curr.copy()
                
    if best_circles is not None and time.time() < deadline - 2:
        lp_r = _solve_lp_radii(best_circles, w, h)
        if lp_r is not None:
            c_lp = best_circles.copy()
            c_lp[:, 2] = lp_r
            if _strictly_valid(c_lp, w, h):
                val_lp = float(np.sum(lp_r))
                if val_lp > best_val:
                    best_val = val_lp
                    best_circles = c_lp.copy()
                    
    return best_val, best_circles

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None
    best_t = 0.5

    t_coarse = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    for t_init in t_coarse:
        if time.time() > deadline - 20:
            break
            
        w = 2.0 * t_init
        h = 2.0 * (1.0 - t_init)
        
        circles_init = _hex_init(w, h)
        
        val, circles = _run_optimization_for_box(w, h, circles_init, deadline)
        
        if circles is not None and val > best_val:
            best_val = val
            best_result = circles
            best_t = t_init

    if best_t > 0 and time.time() < deadline - 30:
        t_fine = np.linspace(best_t - 0.05, best_t + 0.05, 5)
        t_fine = [t for t in t_fine if 0.2 <= t <= 0.8]
        
        for t_init in t_fine:
            if time.time() > deadline - 10:
                break
            
            w = 2.0 * t_init
            h = 2.0 * (1.0 - t_init)
            
            circles_init = _hex_init(w, h)
            
            val, circles = _run_optimization_for_box(w, h, circles_init, deadline)
            
            if circles is not None and val > best_val:
                best_val = val
                best_result = circles
                best_t = t_init

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

    return best_result
