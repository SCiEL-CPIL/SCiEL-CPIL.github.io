"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def softplus(x):
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    def sigmoid(x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    W_BOUNDARY = 100.0
    W_OVERLAP = 50.0

    # We optimize over: w (width), and for each circle (x_i, y_i, r_i)
    # h = 2 - w, perimeter = 2*(w+h) = 4
    # Total variables: 1 + 3*N = 64

    def make_initial(w, h, n_circles):
        """Create a hexagonal-like packing initialization."""
        if n_circles == 0:
            return np.zeros(0)
        
        # Hexagonal packing: rows offset
        # For n circles, find best grid
        best_score = -1
        best_arr = None
        
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows  # ceil division
            # Try this grid
            # Hexagonal: each row offset by 0.5 * spacing_x
            # spacing_x = w / cols (approx)
            # spacing_y = h / rows
            # For hexagonal, vertical spacing = 0.866 * horizontal spacing (for equilateral)
            # But we're in a rectangle, so we just use grid with hex offset
            
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                # Center of cell
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                # Hex offset: shift alternate rows
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                # Clamp
                x = np.clip(x, 0.1 * w, 0.9 * w)
                y = np.clip(y, 0.1 * h, 0.9 * h)
                
                # Radius: half the min of cell width and height, times safety factor
                cell_w = w / cols
                cell_h = h / rows
                r = 0.45 * min(cell_w, cell_h)
                
                xs[i] = x
                ys[i] = y
                rs[i] = r
            
            # Quick validity check
            valid = True
            for i in range(n_circles):
                if xs[i] < rs[i] or xs[i] > w - rs[i]:
                    valid = False
                    break
                if ys[i] < rs[i] or ys[i] > h - rs[i]:
                    valid = False
                    break
            if valid:
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < rs[i] + rs[j]:
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
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(n_circles)))
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            for i in range(n_circles):
                row = i // side
                col = i % side
                xs[i] = (col + 0.5) / side * w
                ys[i] = (row + 0.5) / side * h
                rs[i] = 0.45 / side * min(w, h)
            best_arr = np.concatenate([xs, ys, rs])
        
        return best_arr

    def objective_and_grad(params):
        """Penalty-based objective with analytic gradient. Uses softplus for smoothness."""
        w = params[0]
        w_clipped = np.clip(w, 0.3, 1.7)
        h = 2.0 - w_clipped
        
        circles = params[1:]
        obj_val = 0.0
        grad = np.zeros_like(params)
        
        sum_r = 0.0
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            sum_r += ri
            obj_val -= ri
            grad[3*i+2] -= 1.0

            # Boundary penalties
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

        # Overlap penalties
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            for j in range(i+1, N):
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

    def is_valid(params):
        """Check if configuration is valid."""
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-6:
                return False
            if xi - ri < -1e-5 or xi + ri > w + 1e-5:
                return False
            if yi - ri < -1e-5 or yi + ri > h + 1e-5:
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
                if dist_sq < (ri + rj - 1e-5)**2:
                    return False
        return True

    def get_value(params):
        """Return sum of radii if valid, else -inf."""
        if not is_valid(params):
            return -np.inf
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        # Verify bounding box perimeter
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

    best_value = -np.inf
    best_params = None

    # Try multiple aspect ratios and initializations
    random.seed(42)
    np.random.seed(42)
    
    # Generate a set of candidate initial configurations
    candidates = []
    
    # 1. Square-ish with hexagonal pattern
    for w_try in [0.9, 1.0, 1.1, 0.95, 1.05]:
        h_try = 2.0 - w_try
        arr = make_initial(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    # 2. Random perturbations of hexagonal
    for _ in range(30):
        w_try = random.uniform(0.85, 1.15)
        h_try = 2.0 - w_try
        arr = make_initial(w_try, h_try, N)
        # Perturb
        arr += np.random.randn(len(arr)) * 0.02
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    # 3. Simple grid initialization
    for w_try in [0.8, 0.9, 1.0, 1.1, 1.2]:
        h_try = 2.0 - w_try
        side = 5  # 5x5 = 25, use 21
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side * w_try
            ys[i] = (row + 0.5) / side * h_try
            rs[i] = 0.48 / side * min(w_try, h_try)
        arr = np.concatenate([xs, ys, rs])
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    # Evaluate and improve each candidate
    # We will use SLSQP with analytic gradients for faster and more reliable convergence
    bounds = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2*N) + [(1e-5, 1.0)] * N
    
    for idx, init_params in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        # First, check if valid as-is
        val = get_value(init_params)
        if val > best_value:
            best_value = val
            best_params = init_params.copy()
        
        # Try local optimization using L-BFGS-B (faster)
        try:
            result = minimize(
                objective_and_grad,
                init_params,
                method='L-BFGS-B',
                jac=True, # Indicates that objective returns (f, g)
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            opt_params = result.x
            opt_val = get_value(opt_params)
            if opt_val > best_value:
                best_value = opt_val
                best_params = opt_params.copy()
        except Exception:
            pass
    
    # If we have a valid best, try to refine with more iterations
    if best_params is not None and time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            try:
                # Use SLSQP for final polish as it handles constraints slightly better
                result = minimize(
                    objective_and_grad,
                    best_params,
                    method='SLSQP',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )
                opt_params = result.x
                opt_val = get_value(opt_params)
                if opt_val > best_value:
                    best_value = opt_val
                    best_params = opt_params.copy()
            except Exception:
                break

    # If still no valid solution, create a guaranteed valid one
    if best_params is None or best_value < 0:
        # Fallback: 5x5 grid in a 1x1 square (perimeter 4)
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.48 / side
        return result

    # Extract final result
    w_final = np.clip(best_params[0], 0.1, 1.9)
    h_final = 2.0 - w_final
    circles = best_params[1:]
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
    # Final verification: ensure bounding box perimeter <= 4
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
        # Scale down if needed
        scale = 4.0 / bbox_perim
        # Scale positions and radii relative to center of bbox
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    
    return result
