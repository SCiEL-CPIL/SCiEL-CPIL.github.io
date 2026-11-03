"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def softplus(x):
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    def sigmoid(x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    W_BOUNDARY = 200.0
    W_OVERLAP = 200.0
    W_VARIANCE = 0.5

    def make_hex_init(w, h):
        """Create a hexagonal-like packing initialization for 21 circles."""
        best_score = -1
        best_arr = None
        
        for rows in range(1, 8):
            base = 21 // rows
            extra = 21 % rows
            row_counts = [base + (1 if i < extra else 0) for i in range(rows)]
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            idx = 0
            for row in range(rows):
                n_in_row = row_counts[row]
                if n_in_row == 0:
                    continue
                offset = (w / n_in_row) * 0.5 if row % 2 == 1 else 0
                for col in range(n_in_row):
                    x = offset + (col + 0.5) * (w / n_in_row)
                    y = (row + 0.5) * (h / rows)
                    x = np.clip(x, 0.15 * w, 0.85 * w)
                    y = np.clip(y, 0.15 * h, 0.85 * h)
                    
                    if n_in_row > 1:
                        cell_w = w / n_in_row
                    else:
                        cell_w = w
                    cell_h = h / rows
                    r = 0.42 * min(cell_w, cell_h)
                    
                    xs[idx] = x
                    ys[idx] = y
                    rs[idx] = r
                    idx += 1
            
            valid = True
            for i in range(N):
                if xs[i] < rs[i] or xs[i] > w - rs[i]:
                    valid = False
                    break
                if ys[i] < rs[i] or ys[i] > h - rs[i]:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_arr = np.concatenate([xs, ys, rs])
        
        if best_arr is None:
            side = 5
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xs[i] = (col + 0.5) / side * w
                ys[i] = (row + 0.5) / side * h
                rs[i] = 0.45 / side * min(w, h)
            best_arr = np.concatenate([xs, ys, rs])
        
        return best_arr

    def make_grid_init(w, h, rows, cols):
        total = rows * cols
        if total < N:
            rows = int(np.ceil(N / cols))
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            row = i // cols
            col = i % cols
            xs[i] = (col + 0.5) / cols * w
            ys[i] = (row + 0.5) / rows * h
            cell_w = w / cols
            cell_h = h / rows
            rs[i] = 0.45 * min(cell_w, cell_h)
        
        return np.concatenate([xs, ys, rs])

    def get_delaunay_neighbors(params):
        circles = params[1:]
        xs = circles[0::3]
        ys = circles[1::3]
        
        if len(xs) < 3:
            return []
        
        points = np.column_stack([xs, ys])
        try:
            tri = Delaunay(points)
            neighbors = set()
            for simplex in tri.simplices:
                for i in simplex:
                    for j in simplex:
                        if i != j:
                            neighbors.add((min(i, j), max(i, j)))
            return list(neighbors)
        except Exception:
            return [(i, j) for i in range(N) for j in range(i+1, N)]

    def objective_and_grad(params, neighbor_list=None):
        w = params[0]
        w_clipped = np.clip(w, 0.3, 1.7)
        h = 2.0 - w_clipped
        
        circles = params[1:]
        obj_val = 0.0
        grad = np.zeros_like(params)
        
        # Sum of radii (negative for minimization)
        for i in range(N):
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            obj_val -= ri
            grad[3*i+2] -= 1.0
        
        # Variance penalty to encourage equal radii
        rs = circles[2::3]
        mean_r = np.mean(rs)
        var_r = np.var(rs)
        obj_val += W_VARIANCE * var_r
        for i in range(N):
            grad[3*i+2] += W_VARIANCE * 2.0 * (circles[3*i+2] - mean_r) / N

        # Boundary penalties
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            
            v_left = ri - xi
            if v_left > -20:
                sp_grad = sigmoid(v_left)
                obj_val += W_BOUNDARY * softplus(v_left)
                grad[3*i] -= W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad
            
            v_right = xi + ri - w_clipped
            if v_right > -20:
                sp_grad = sigmoid(v_right)
                obj_val += W_BOUNDARY * softplus(v_right)
                grad[3*i] += W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad
                
            v_bot = ri - yi
            if v_bot > -20:
                sp_grad = sigmoid(v_bot)
                obj_val += W_BOUNDARY * softplus(v_bot)
                grad[3*i+1] -= W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad

            v_top = yi + ri - h
            if v_top > -20:
                sp_grad = sigmoid(v_top)
                obj_val += W_BOUNDARY * softplus(v_top)
                grad[3*i+1] += W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad

        # Overlap penalties (using neighbor list if provided)
        if neighbor_list is None:
            neighbor_list = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        for i, j in neighbor_list:
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            xj = circles[3*j]
            yj = circles[3*j+1]
            rj = circles[3*j+2]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            min_dist = ri + rj
            
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                v = min_dist - dist
                if v > -20:
                    sp_grad = sigmoid(v)
                    obj_val += W_OVERLAP * softplus(v)
                    
                    coeff = W_OVERLAP * sp_grad / dist
                    grad[3*i] -= coeff * dx
                    grad[3*i+1] -= coeff * dy
                    grad[3*i+2] += W_OVERLAP * sp_grad
                    
                    grad[3*j] += coeff * dx
                    grad[3*j+1] += coeff * dy
                    grad[3*j+2] += W_OVERLAP * sp_grad

        # Gradient for w
        grad_w = 0.0
        for i in range(N):
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            v_top = yi + ri - h
            if v_top > -20:
                grad_w += W_BOUNDARY * sigmoid(v_top)
        grad[0] = grad_w

        return obj_val, grad

    def is_valid(params, tol=1e-5):
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-6:
                return False
            if xi - ri < -tol or xi + ri > w + tol:
                return False
            if yi - ri < -tol or yi + ri > h + tol:
                return False
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            for j in range(i+1, N):
                xj = circles[3*j]
                yj = circles[3*j+1]
                rj = circles[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                if dist_sq < (ri + rj - tol)**2:
                    return False
        return True

    def get_value(params):
        if not is_valid(params):
            return -np.inf
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        xs = circles[0::3]
        ys = circles[1::3]
        rs = circles[2::3]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_perim = 2 * (bbox_w + bbox_h)
        if bbox_perim > 4.0 + 1e-5:
            return -np.inf
        return np.sum(rs)

    def polish(params, max_iter=200):
        best_params = params.copy()
        best_val = get_value(params)
        step_size = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            obj, grad = objective_and_grad(params)
            
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-8:
                break
            
            improved = False
            for trial_step in [step_size, step_size * 0.5, step_size * 2]:
                new_params = params - trial_step * grad
                new_params[0] = np.clip(new_params[0], 0.3, 1.7)
                for i in range(N):
                    new_params[3*i+2] = np.clip(new_params[3*i+2], 1e-5, 1.0)
                
                new_val = get_value(new_params)
                if new_val > best_val + 1e-10:
                    best_val = new_val
                    best_params = new_params.copy()
                    params = new_params.copy()
                    step_size = trial_step * 1.1
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-6:
                    break
        
        return best_params, best_val

    best_value = -np.inf
    best_params = None

    random.seed(42)
    np.random.seed(42)
    
    candidates = []
    
    for w_try in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        h_try = 2.0 - w_try
        arr = make_hex_init(w_try, h_try)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    for w_try in [0.9, 1.0, 1.1]:
        h_try = 2.0 - w_try
        for rows in [4, 5, 6]:
            cols = int(np.ceil(N / rows))
            arr = make_grid_init(w_try, h_try, rows, cols)
            params = np.zeros(1 + 3*N)
            params[0] = w_try
            params[1:] = arr
            candidates.append(params.copy())
    
    for _ in range(20):
        w_try = random.uniform(0.85, 1.15)
        h_try = 2.0 - w_try
        arr = make_hex_init(w_try, h_try)
        arr += np.random.randn(len(arr)) * 0.01
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    bounds = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2*N) + [(1e-5, 1.0)] * N
    
    for idx, init_params in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        val = get_value(init_params)
        if val > best_value:
            best_value = val
            best_params = init_params.copy()
        
        neighbors = get_delaunay_neighbors(init_params)
        
        try:
            def obj_with_neighbors(params):
                return objective_and_grad(params, neighbors)
            
            result = minimize(
                obj_with_neighbors,
                init_params,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            opt_params = result.x
            opt_val = get_value(opt_params)
            if opt_val > best_value:
                best_value = opt_val
                best_params = opt_params.copy()
        except Exception:
            pass
        
        if best_params is not None:
            polished_params, polished_val = polish(best_params, max_iter=100)
            if polished_val > best_value:
                best_value = polished_val
                best_params = polished_params.copy()

    if best_params is not None and time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            polished_params, polished_val = polish(best_params, max_iter=150)
            if polished_val > best_value:
                best_value = polished_val
                best_params = polished_params.copy()

    if best_params is None or best_value < 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.45 / side
        return result

    w_final = np.clip(best_params[0], 0.1, 1.9)
    h_final = 2.0 - w_final
    circles = best_params[1:]
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_perim = 2 * (bbox_w + bbox_h)
    
    if bbox_perim > 4.0 + 1e-5:
        scale = 4.0 / bbox_perim
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    
    return result
