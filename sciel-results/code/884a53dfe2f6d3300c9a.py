import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Initial Guess: Grid
    # A 5x5 grid has 25 cells. We need 26.
    # Let's try a 5x6 grid or similar, or just randomize a 5x5 and add one.
    # Better: Use a grid of size ceil(sqrt(N)) x ceil(sqrt(N)) and take the first N.
    # For N=26, 5x5=25, 6x6=36.
    # Let's use a 5x6 grid arrangement to spread them out.
    side_x = 5
    side_y = 6
    positions = []
    for i in range(side_x):
        for j in range(side_y):
            x = (i + 0.5) / side_x
            y = (j + 0.5) / side_y
            positions.append([x, y])
    
    # Take first N
    positions = positions[:N]
    positions = np.array(positions)
    
    # Initial radii: small enough to fit
    # Max possible radius is roughly 1/side_x
    r_init = 0.5 / side_x - 0.01
    radii = np.full(N, r_init)
    
    # Variable vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = positions[i, 0]
        x0[3*i+1] = positions[i, 1]
        x0[3*i+2] = radii[i]
    
    def unpack(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        return xs, ys, rs

    # Objective for L-BFGS-B: Maximize sum of radii subject to soft constraints
    # Minimize -sum(rs) + penalty
    def objective(v):
        xs, ys, rs = unpack(v)
        
        # Boundary penalty
        # x - r >= 0 => r - x <= 0
        # 1 - x - r >= 0 => x + r - 1 <= 0
        viol_x_left = np.maximum(0, rs - xs)
        viol_x_right = np.maximum(0, xs + rs - 1)
        viol_y_bottom = np.maximum(0, rs - ys)
        viol_y_top = np.maximum(0, ys + rs - 1)
        
        boundary_viol = np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + \
                        np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2)
        
        # Overlap penalty
        # dist(i,j) >= ri + rj
        # dist^2 - (ri+rj)^2 >= 0
        # We want to penalize when dist < ri + rj
        overlap_viol = 0.0
        # Efficient loop for N=26
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                req_dist = rs[i] + rs[j]
                if dist < req_dist:
                    overlap_viol += (req_dist - dist)**2
        
        # Sum of radii to maximize (minimize negative)
        sum_r = np.sum(rs)
        
        # Penalty weight
        # Start with high penalty, can be reduced if stuck, but usually fixed is fine for small N
        penalty_weight = 10.0
        
        return -sum_r + penalty_weight * (boundary_viol + overlap_viol)

    # Gradient for L-BFGS-B (numerical is slow, let's use finite difference or analytical if possible)
    # For N=26, analytical gradient is better.
    def gradient(v):
        xs, ys, rs = unpack(v)
        grad = np.zeros_like(v)
        
        # Boundary gradients
        # d/dx (max(0, r-x)^2) = -2 max(0, r-x)
        # d/dx (max(0, x+r-1)^2) = 2 max(0, x+r-1)
        # d/dr (max(0, r-x)^2) = 2 max(0, r-x)
        # d/dr (max(0, x+r-1)^2) = 2 max(0, x+r-1)
        
        viol_x_left = np.maximum(0, rs - xs)
        viol_x_right = np.maximum(0, xs + rs - 1)
        viol_y_bottom = np.maximum(0, rs - ys)
        viol_y_top = np.maximum(0, ys + rs - 1)
        
        for i in range(N):
            grad[3*i]   = -2 * viol_x_left[i] + 2 * viol_x_right[i]  # dx
            grad[3*i+1] = -2 * viol_y_bottom[i] + 2 * viol_y_top[i]  # dy
            grad[3*i+2] = -1 + 2 * (viol_x_left[i] + viol_x_right[i] + viol_y_bottom[i] + viol_y_top[i]) # dr (from -sum(r))
            
        # Overlap gradients
        # Penalty term: (ri + rj - dist)^2
        # d/dxi = -2(ri+rj-dist) * (xi-xj)/dist
        # d/dri = 2(ri+rj-dist)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                req_dist = rs[i] + rs[j]
                diff = req_dist - dist
                if diff > 0:
                    factor = 2 * diff
                    grad[3*i]   += -factor * dx / dist
                    grad[3*i+1] += -factor * dy / dist
                    grad[3*i+2] += factor
                    
                    grad[3*j]   += factor * dx / dist
                    grad[3*j+1] += factor * dy / dist
                    grad[3*j+2] += factor
        
        # Apply penalty weight
        grad *= 10.0
        
        return grad

    # Stage 1: L-BFGS-B
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    # Reorder bounds to match vector [x1,y1,r1, x2,y2,r2...]
    bounds_vec = []
    for i in range(N):
        bounds_vec.append((0, 1)) # x
        bounds_vec.append((0, 1)) # y
        bounds_vec.append((0, 0.5)) # r

    try:
        res1 = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds_vec, 
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        x_opt = res1.x
    except Exception:
        x_opt = x0

    if time.time() > deadline:
        # Fallback to initial if too slow
        pass
    else:
        # Stage 2: SLSQP Polish
        # Define constraints for SLSQP
        def constraints(v):
            cons = []
            xs, ys, rs = unpack(v)
            
            # Boundary constraints:
            # x - r >= 0
            # 1 - x - r >= 0
            # y - r >= 0
            # 1 - y - r >= 0
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
            
            # Overlap constraints:
            # dist(i,j) - ri - rj >= 0
            for i in range(N):
                for j in range(i+1, N):
                    def overlap_fun(v, i=i, j=j):
                        xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                        xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                        d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                        return d - ri - rj
                    cons.append({'type': 'ineq', 'fun': overlap_fun})
            return cons

        # Objective for SLSQP: Just -sum(r)
        def obj_slsqp(v):
            rs = v[2::3]
            return -np.sum(rs)

        # Gradient for SLSQP
        def grad_slsqp(v):
            grad = np.zeros_like(v)
            grad[2::3] = -1.0
            return grad

        try:
            res2 = minimize(obj_slsqp, x_opt, method='SLSQP', jac=grad_slsqp, bounds=bounds_vec, 
                            constraints=constraints(x_opt), 
                            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
            if res2.success and res2.fun < obj_slsqp(x_opt):
                x_opt = res2.x
        except Exception:
            pass

    # Extract final result
    xs = x_opt[0::3]
    ys = x_opt[1::3]
    rs = x_opt[2::3]
    
    # Post-processing: Ensure strict validity
    # Check overlaps and boundaries, shrink radii if necessary
    for i in range(N):
        # Boundary check
        max_r_boundary = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
        if rs[i] > max_r_boundary:
            rs[i] = max_r_boundary * (1 - 1e-6)
            
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                # Scale down both radii proportionally to fit
                # Need dist >= r_i' + r_j'
                # r_i' = r_i * scale, r_j' = r_j * scale
                # scale = dist / (r_i + r_j)
                scale = dist / min_dist
                rs[i] *= scale * (1 - 1e-6)
                rs[j] *= scale * (1 - 1e-6)

    result = np.column_stack((xs, ys, rs))
    return result

# EVOLVE-BLOCK-START
# (The code above is the full implementation. The prompt asks to keep the function names.
# I will wrap the logic inside the construct function as required by the seed structure, 
# or just provide the full file. The seed had EVOLVE-BLOCK-START/END. 
# I will place the code inside that block if possible, or just replace the whole file content 
# as the instruction says "Rewrite the program from scratch".)
# EVOLVE-BLOCK-END
