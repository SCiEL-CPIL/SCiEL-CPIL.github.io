import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    if pos_r.shape[0] != n_circles:
        return False
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        if not moved:
            break
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
    return np.column_stack((pos, rad))

def _generate_hex_init(w, h, n_circles):
    circles = np.zeros((n_circles, 3))
    r_est = np.sqrt((w * h) / (n_circles * 1.1))
    rows = []
    y = r_est
    row_idx = 0
    while y < h:
        x = r_est if row_idx % 2 == 0 else 2*r_est
        while x < w:
            rows.append([x, y, r_est])
            x += 2*r_est
        y += np.sqrt(3) * r_est
        row_idx += 1
    if len(rows) < n_circles:
        return _generate_grid_init(w, h, n_circles, 5, 5)
    for i in range(n_circles):
        circles[i] = rows[i]
    return circles

def _generate_grid_init(w, h, n_circles, cols, rows):
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
    return circles

def _softplus(x, beta=50.0):
    """Smooth approximation of max(0, x): log(1 + exp(beta*x))/beta"""
    # Numerically stable softplus
    beta_x = beta * x
    # For large positive beta_x, softplus ≈ x; for large negative, ≈ 0
    result = np.where(
        beta_x > 20,
        x,
        np.log1p(np.exp(np.clip(beta_x, -50, 20))) / beta
    )
    return result

def _tighten_positions(circles, w, h, deadline):
    """
    Optimize positions to minimize soft-plus of all clearances.
    Uses a smooth surrogate for the nonsmooth max(0, slack) objective.
    """
    if time.time() > deadline - 12.0:
        return circles

    n = len(circles)
    pos0 = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    # Precompute pair indices
    pair_i, pair_j = np.triu_indices(n, k=1)
    
    def objective(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        # Wall slacks (4 per circle)
        wall_slacks = np.concatenate([
            pos[:, 0] - rad,
            w - pos[:, 0] - rad,
            pos[:, 1] - rad,
            h - pos[:, 1] - rad
        ])
        # Pair slacks
        dx = pos[pair_i, 0] - pos[pair_j, 0]
        dy = pos[pair_i, 1] - pos[pair_j, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        pair_slacks = dists - (rad[pair_i] + rad[pair_j])
        
        all_slacks = np.concatenate([wall_slacks, pair_slacks])
        # Smooth surrogate: sum of softplus(slack)
        return np.sum(_softplus(all_slacks, beta=50.0))
    
    def jac(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        grad = np.zeros_like(pos_flat)
        
        # Wall contributions (gradient of softplus is sigmoid-like)
        def dsoftplus(slack):
            bx = 50.0 * slack
            # d/dx log(1+exp(bx))/b = exp(bx)/(1+exp(bx)) = sigmoid(bx)
            return np.where(bx > 20, 1.0, 1.0 / (1.0 + np.exp(-np.clip(bx, -50, 20))))
        
        # x-walls
        grad[0::2] += dsoftplus(pos[:, 0] - rad)  # x - r
        grad[0::2] -= dsoftplus(w - pos[:, 0] - rad)  # w - x - r
        # y-walls
        grad[1::2] += dsoftplus(pos[:, 1] - rad)  # y - r
        grad[1::2] -= dsoftplus(h - pos[:, 1] - rad)  # h - y - r
        
        # Pair contributions
        dx = pos[pair_i, 0] - pos[pair_j, 0]
        dy = pos[pair_i, 1] - pos[pair_j, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        pair_slacks = dists - (rad[pair_i] + rad[pair_j])
        dsl = dsoftplus(pair_slacks)
        
        # d/dx_i of dist = dx/dist
        for k in range(len(pair_i)):
            if dists[k] > 1e-12:
                nx = dx[k] / dists[k]
                ny = dy[k] / dists[k]
                # Gradient w.r.t. pos[i] is +nx, +ny (since dist = |pos_i - pos_j|)
                # But we want to minimize slack = dist - (r_i+r_j), so d(slack)/d(pos_i) = (pos_i-pos_j)/dist
                # d(objective)/d(pos_i) = dsoftplus * d(slack)/d(pos_i)
                idx_i = pair_i[k] * 2
                idx_j = pair_j[k] * 2
                grad[idx_i] += dsl[k] * nx
                grad[idx_i + 1] += dsl[k] * ny
                grad[idx_j] -= dsl[k] * nx
                grad[idx_j + 1] -= dsl[k] * ny
        
        return grad
    
    bounds = []
    for i in range(n):
        bounds.append((rad[i], w - rad[i]))
        bounds.append((rad[i], h - rad[i]))
    
    try:
        res = minimize(
            objective,
            pos0.flatten(),
            jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 40}
        )
        
        if res.success or res.fun < objective(pos0.flatten()):
            new_pos = res.x.reshape(-1, 2)
            new_circles = np.column_stack((new_pos, rad))
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                # Try to fix
                fixed = _resolve_collisions(new_circles, w, h, max_iters=200)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass
        
    return circles

def _slsqp_optimize(circles, w, h, deadline, maxiter=200, ftol=1e-9):
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        r = x[2::3]
        return -np.sum(r)

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol}
        )

        if res.success:
            new_circles = res.x.reshape(-1, 3)
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _polish(circles, w, h, deadline):
    """High-precision polish: run SLSQP with tighter tolerances on the best result."""
    if time.time() > deadline - 3.0:
        return circles
    result = _slsqp_optimize(circles, w, h, deadline, maxiter=300, ftol=1e-11)
    if result is not None:
        return result
    return circles

def _solve_aspect(w, h, deadline):
    best_init = None
    best_init_score = 0.0
    
    hex_init = _generate_hex_init(w, h, N)
    relaxed = _resolve_collisions(hex_init, w, h, max_iters=200)
    if _validate_solution(relaxed, w, h, N):
        score = np.sum(relaxed[:, 2])
        if score > best_init_score:
            best_init_score = score
            best_init = relaxed

    grid_configs = [(4, 5), (5, 4), (5, 5), (3, 7), (7, 3), (4, 6), (6, 4)]
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed

    if best_init is None:
        return None

    if time.time() < deadline - 15.0:
        tightened = _tighten_positions(best_init, w, h, deadline)
        if tightened is not None:
            best_init = tightened

    if time.time() < deadline - 5.0:
        optimized = _slsqp_optimize(best_init, w, h, deadline)
        if optimized is not None:
            return optimized

    return best_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_result = None
    best_score = 0.0
    best_wh = (1.0, 1.0)
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 2.0, 0.55, 1.3, 0.65, 1.5, 0.75]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        result = _solve_aspect(w, h, deadline)
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                best_wh = (w, h)

    # Final polish on the best result
    if best_result is not None and time.time() < deadline - 3.0:
        w, h = best_wh
        polished = _polish(best_result, w, h, deadline)
        if polished is not None and _validate_solution(polished, w, h, N):
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_result = polished
                best_score = score

    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _generate_grid_init(w, h, N, 5, 5)
        best_result = _resolve_collisions(best_result, w, h, max_iters=200)
        if not _validate_solution(best_result, w, h, N):
            for i in range(N):
                x, y, r = best_result[i]
                min_d = min(x, w-x, y, h-y)
                best_result[i, 2] = min(r, min_d - 1e-4)
            best_result = _resolve_collisions(best_result, w, h, max_iters=200)

    return best_result
