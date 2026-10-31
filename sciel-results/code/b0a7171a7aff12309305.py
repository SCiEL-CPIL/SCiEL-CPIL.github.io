"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    """Pack 21 circles in a perimeter-4 rectangle to maximize sum of radii."""
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit

    best_val = 0.0
    best_result = None

    # Perimeter 4 => 2*(w+h) = 4 => w+h = 2
    # We test a range of aspect ratios w/h
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 1.5, 0.67]

    def get_hex_init(w, h, n, seed=0):
        """
        Generates initial positions for n circles in a w x h rectangle.
        Uses a hexagonal lattice pattern.
        """
        np.random.seed(seed)
        
        # Estimate spacing based on area
        # Approximate density of hex packing is ~0.9069
        # Area of rect = w*h
        # Area of n circles = n * pi * r^2
        # n * pi * r^2 <= 0.9 * w * h
        # r <= sqrt(0.9 * w * h / (n * pi))
        r_est = math.sqrt(0.9 * w * h / (n * math.pi))
        
        # Hex spacing
        dx = 2.0 * r_est
        dy = math.sqrt(3.0) * r_est
        
        positions = []
        row = 0
        while len(positions) < n:
            for col in range(int(w / dx) + 2):
                if len(positions) >= n:
                    break
                x = (col + 0.5) * dx
                # Stagger odd rows
                if row % 2 == 1:
                    x += dx / 2.0
                y = (row + 0.5) * dy
                
                # Check if inside bounds (with margin)
                if x < r_est or x > w - r_est:
                    continue
                if y < r_est or y > h - r_est:
                    continue
                
                # Add jitter
                jitter_x = np.random.uniform(-0.05, 0.05) * dx
                jitter_y = np.random.uniform(-0.05, 0.05) * dy
                x += jitter_x
                y += jitter_y
                
                # Clamp
                x = np.clip(x, r_est, w - r_est)
                y = np.clip(y, r_est, h - r_est)
                
                positions.append([x, y])
            row += 1
            
        # If we didn't get enough, fallback to grid
        if len(positions) < n:
            cols = int(np.round(np.sqrt(n * w / h)))
            if cols < 1: cols = 1
            rows = int(np.ceil(n / cols))
            while cols * rows < n:
                if w > h: cols += 1
                else: rows += 1
            positions = []
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= n: break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    x += np.random.uniform(-0.1, 0.1) * (w/cols)
                    y += np.random.uniform(-0.1, 0.1) * (h/rows)
                    x = np.clip(x, 0.05, w-0.05)
                    y = np.clip(y, 0.05, h-0.05)
                    positions.append([x, y])
                    idx += 1
            positions = positions[:n]
            
        return np.array(positions[:n])

    def evaluate_and_polish(w, h, init_pos, max_time_left):
        """
        Runs penalty-based optimization followed by SLSQP polish.
        """
        if time.time() > t_start + max_time_left:
            return None, -np.inf

        positions = init_pos
        # Initial radii
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
        
        r0 = 0.4 * min_dist
        radii = np.full(N, r0)
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]

        # Penalty Method Objective
        # Maximize sum(r) - lambda * (violations)^2
        # We want to minimize -sum(r) + lambda * (violations)^2
        
        def penalty_obj(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            # 1. Boundary violations
            # x - r >= 0 => viol = max(0, r - x)
            # w - x - r >= 0 => viol = max(0, x + r - w)
            # y - r >= 0 => viol = max(0, r - y)
            # h - y - r >= 0 => viol = max(0, y + r - h)
            
            viol_left = np.maximum(0, r - x)
            viol_right = np.maximum(0, x + r - w)
            viol_bottom = np.maximum(0, r - y)
            viol_top = np.maximum(0, y + r - h)
            
            boundary_viol = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
            
            # 2. Overlap violations
            # dist - r_i - r_j >= 0 => viol = max(0, r_i + r_j - dist)
            overlap_viol = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    viol = r[i] + r[j] - dist
                    if viol > 0:
                        overlap_viol += viol**2
            
            # 3. Negative radius
            neg_r_viol = np.sum(np.maximum(0, -r)**2)
            
            total_viol = boundary_viol + overlap_viol + neg_r_viol
            
            # Objective: -sum(r) + penalty
            # We use an adaptive penalty or fixed large penalty.
            # Fixed penalty is simpler and robust for L-BFGS-B
            penalty_lambda = 100.0
            return -np.sum(r) + penalty_lambda * total_viol

        # Bounds for L-BFGS-B
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w)) # x
            bounds.append((0, h)) # y
            bounds.append((0, min(w, h)/2)) # r

        try:
            # Step 1: Penalty Optimization
            res1 = minimize(
                penalty_obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            v = res1.x
            x_res = v[0::3]
            y_res = v[1::3]
            r_res = v[2::3]
            
            # Check if it's a valid solution (low violation)
            # Calculate violation
            viol_check = 0.0
            if np.any(x_res < r_res - 1e-4) or np.any(x_res > w - r_res + 1e-4) or \
               np.any(y_res < r_res - 1e-4) or np.any(y_res > h - r_res + 1e-4):
                viol_check = 1.0
            else:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                        if dist < r_res[i] + r_res[j] - 1e-4:
                            viol_check = 1.0
                            break
                    if viol_check: break
            
            if viol_check == 0.0:
                # Step 2: Polish with SLSQP
                def objective(v):
                    return -np.sum(v[2::3])
                
                def cons_inside(v):
                    x = v[0::3]
                    y = v[1::3]
                    r = v[2::3]
                    return np.concatenate([
                        x - r,
                        w - x - r,
                        y - r,
                        h - y - r
                    ])
                
                def cons_non_overlap(v):
                    x = v[0::3]
                    y = v[1::3]
                    r = v[2::3]
                    cons = []
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            cons.append(dist - r[i] - r[j])
                    return np.array(cons)
                
                def cons_positive_r(v):
                    return v[2::3]
                
                constraints = [
                    {'type': 'ineq', 'fun': cons_inside},
                    {'type': 'ineq', 'fun': cons_non_overlap},
                    {'type': 'ineq', 'fun': cons_positive_r}
                ]
                
                res2 = minimize(
                    objective, v, method='SLSQP',
                    constraints=constraints,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                
                if res2.success:
                    v = res2.x
                    x_res = v[0::3]
                    y_res = v[1::3]
                    r_res = v[2::3]
                
                # Final Validation
                valid = True
                if np.any(r_res <= 0):
                    valid = False
                else:
                    if np.any(x_res < r_res - 1e-5) or np.any(x_res > w - r_res + 1e-5):
                        valid = False
                    if np.any(y_res < r_res - 1e-5) or np.any(y_res > h - r_res + 1e-5):
                        valid = False
                    else:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                                if dist < r_res[i] + r_res[j] - 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    total_r = np.sum(r_res)
                    return np.column_stack((x_res, y_res, r_res)).copy(), total_r
            else:
                # Even if penalty solution is invalid, maybe it's close?
                # Try SLSQP anyway from this point? No, SLSQP might fail if start is bad.
                # But let's check if we can fix it with a few SLSQP iterations.
                pass
                
        except Exception:
            pass

        return None, -np.inf

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            positions = get_hex_init(w, h, N, seed=restart*10+int(ar*100))
            
            result, val = evaluate_and_polish(w, h, positions, deadline - time.time())
            
            if result is not None and val > best_val:
                best_val = val
                best_result = result

    # Fallback: Simple grid if optimization failed
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
        best_val = np.sum(best_result[:, 2])

    return best_result

# EVOLVE-BLOCK-END
