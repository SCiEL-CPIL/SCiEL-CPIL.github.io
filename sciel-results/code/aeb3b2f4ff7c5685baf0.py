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

    def make_initial(w, h, n_circles, seed):
        """Create a hexagonal-like packing initialization."""
        rng = np.random.RandomState(seed)
        if n_circles == 0:
            return np.zeros(0)
        
        best_arr = None
        best_score = -1
        
        # Try different grid configurations
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows
            
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                
                # Center of cell
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                
                # Hex offset
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                    
                # Clamp to ensure inside
                x = np.clip(x, 0.1 * w, 0.9 * w)
                y = np.clip(y, 0.1 * h, 0.9 * h)
                
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
        
        # Add small random perturbation to help escape local minima
        if seed > 0:
            best_arr[1::3] += rng.randn(N) * 0.01 * w # x
            best_arr[2::3] += rng.randn(N) * 0.01 * h # y
            # Ensure radii don't become negative or too large
            best_arr[2::3] = np.clip(best_arr[2::3], 0.01, 1.0)
            
        return best_arr

    def objective_and_grad(params, w, h, penalty_weight):
        """
        Objective: Minimize -sum(r) + penalty * (boundary_violation + overlap_violation)
        params: [x1, y1, r1, x2, y2, r2, ...]
        """
        circles = params
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
                obj_val += penalty_weight * softplus(v_left)
                grad[3*i] -= penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad
            
            v_right = xi + ri - w
            if v_right > -20:
                sp_grad = sigmoid(v_right)
                obj_val += penalty_weight * softplus(v_right)
                grad[3*i] += penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad
                
            v_bot = ri - yi
            if v_bot > -20:
                sp_grad = sigmoid(v_bot)
                obj_val += penalty_weight * softplus(v_bot)
                grad[3*i+1] -= penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad

            v_top = yi + ri - h
            if v_top > -20:
                sp_grad = sigmoid(v_top)
                obj_val += penalty_weight * softplus(v_top)
                grad[3*i+1] += penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad

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
                        obj_val += penalty_weight * softplus(v)
                        
                        coeff = penalty_weight * sp_grad / dist
                        grad[3*i] -= coeff * dx
                        grad[3*i+1] -= coeff * dy
                        grad[3*i+2] += penalty_weight * sp_grad
                        
                        grad[3*j] += coeff * dx
                        grad[3*j+1] += coeff * dy
                        grad[3*j+2] += penalty_weight * sp_grad

        return obj_val, grad

    def is_valid(params, w, h):
        """Check if configuration is valid."""
        circles = params
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

    def get_value(params, w, h):
        """Return sum of radii if valid, else -inf."""
        if not is_valid(params, w, h):
            return -np.inf
        return np.sum(params[2::3])

    best_value = -np.inf
    best_params = None
    best_w = 1.0

    random.seed(42)
    np.random.seed(42)

    # Scan aspect ratios
    # w + h = 2
    # w in [0.5, 1.5]
    w_list = np.linspace(0.7, 1.3, 13)
    
    # We will run multiple iterations for each w, increasing penalty weight
    # This is "penalty annealing"
    
    for w in w_list:
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        
        # Initializations
        inits = []
        # 1. Hexagonal
        inits.append(make_initial(w, h, N, 0))
        # 2. Randomized grids
        for seed in [1, 2, 3]:
             inits.append(make_initial(w, h, N, seed))
             
        # Try each initialization
        for init in inits:
            if time.time() > deadline - 5:
                break
                
            current_params = init.copy()
            current_w = w
            current_h = h
            
            # Penalty Annealing
            # Start with low penalty to find a good "shape", then increase to enforce constraints
            penalties = [10, 50, 100, 500, 1000]
            
            for p in penalties:
                if time.time() > deadline - 2:
                    break
                    
                # Define objective for this penalty
                def obj_func(x):
                    return objective_and_grad(x, current_w, current_h, p)
                
                bounds = [(0, current_w)] * N + [(0, current_h)] * N + [(0, min(current_w, current_h)/2)] * N
                # Note: L-BFGS-B bounds are (min, max) for each variable.
                # Variables are x1..xN, y1..yN, r1..rN
                # x_i in [0, w], y_i in [0, h], r_i in [0, min(w,h)/2]
                
                try:
                    result = minimize(
                        obj_func,
                        current_params,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    current_params = result.x
                except Exception:
                    pass
            
            # Check validity and update best
            val = get_value(current_params, current_w, current_h)
            if val > best_value:
                best_value = val
                best_params = current_params.copy()
                best_w = current_w

    # If we have a valid best, try to refine with more iterations and slightly different w
    if best_params is not None and time.time() < deadline - 10:
        # Refinement loop
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            # Try slightly different w around best_w
            w_new = best_w + np.random.uniform(-0.05, 0.05)
            w_new = np.clip(w_new, 0.5, 1.5)
            h_new = 2.0 - w_new
            
            # Rescale current best to new aspect ratio? 
            # Or just start from a new init?
            # Starting from new init is safer.
            init = make_initial(w_new, h_new, N, int(np.random.rand()*1000))
            
            current_params = init.copy()
            penalties = [100, 500, 1000]
            
            for p in penalties:
                if time.time() > deadline - 2:
                    break
                def obj_func(x):
                    return objective_and_grad(x, w_new, h_new, p)
                
                bounds = [(0, w_new)] * N + [(0, h_new)] * N + [(0, min(w_new, h_new)/2)] * N
                
                try:
                    result = minimize(
                        obj_func,
                        current_params,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    current_params = result.x
                except Exception:
                    pass
            
            val = get_value(current_params, w_new, h_new)
            if val > best_value:
                best_value = val
                best_params = current_params.copy()
                best_w = w_new

    # If still no valid solution, create a guaranteed valid one
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
            result[i, 2] = 0.48 / side
        return result

    # Extract final result
    w_final = best_w
    h_final = 2.0 - w_final
    circles = best_params
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
    # Final verification: ensure bounding box perimeter <= 4
    # The optimization ensures circles are in [0, w] x [0, h], so bounding box is at most w x h.
    # Perimeter is 2*(w+h) = 4.
    # However, if circles don't touch boundaries, the *minimal* bounding box is smaller, which is fine.
    # We just need to ensure the *actual* bounding box of the circles has perimeter <= 4.
    
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
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    
    return result
