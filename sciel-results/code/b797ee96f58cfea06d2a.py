"""Circle packing in unit square n=32.
Strategy:
1. Initialize with a grid pattern.
2. Use SLSQP to maximize sum of radii subject to non-overlap and boundary constraints.
3. Use a penalty-based objective to handle the non-smooth min-distance constraint.
4. Polish with tighter tolerances.
5. Deadline guard to stay within 150s.
"""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial positions: grid layout
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for idx in range(N):
        i = idx % side
        j = idx // side
        positions[idx, 0] = (i + 0.5) / side
        positions[idx, 1] = (j + 0.5) / side
    
    # Initial radii: small
    r0 = 0.5 / side - 0.01
    radii = np.full(N, r0)
    
    # Variables: x, y, r for each circle -> 3*N variables
    x0 = np.concatenate([positions, radii])
    
    # Objective: maximize sum of radii -> minimize -sum(radii)
    # Constraints:
    # 1. For each pair (i,j): dist(i,j) >= r_i + r_j
    # 2. For each circle i: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    # Use penalty method for smoothness
    # Penalty weight for overlaps
    penalty_weight = 500.0
    
    def objective_and_grad(x):
        positions = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        # Objective: -sum(radii)
        obj = -np.sum(radii)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += penalty_weight * (min_dist - dist)**2
        
        # Penalty for boundary violations
        for i in range(N):
            xi, yi = positions[i]
            ri = radii[i]
            # Distance to each edge
            d_left = xi
            d_right = 1.0 - xi
            d_bottom = yi
            d_top = 1.0 - yi
            min_edge_dist = min(d_left, d_right, d_bottom, d_top)
            if ri > min_edge_dist:
                penalty += penalty_weight * (ri - min_edge_dist)**2
        
        total = obj + penalty
        
        # Gradient (approximate)
        grad = np.zeros_like(x)
        eps = 1e-7
        for k in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[k] += eps
            xm[k] -= eps
            # Compute objective with penalty for xp and xm
            pos_p = xp[:2*N].reshape(N, 2)
            rad_p = xp[2*N:]
            obj_p = -np.sum(rad_p)
            pen_p = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos_p[i, 0] - pos_p[j, 0]
                    dy = pos_p[i, 1] - pos_p[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    min_dist = rad_p[i] + rad_p[j]
                    if dist < min_dist:
                        pen_p += penalty_weight * (min_dist - dist)**2
            for i in range(N):
                xi, yi = pos_p[i]
                ri = rad_p[i]
                min_edge_dist = min(xi, 1.0-xi, yi, 1.0-yi)
                if ri > min_edge_dist:
                    pen_p += penalty_weight * (ri - min_edge_dist)**2
            val_p = obj_p + pen_p
            
            pos_m = xm[:2*N].reshape(N, 2)
            rad_m = xm[2*N:]
            obj_m = -np.sum(rad_m)
            pen_m = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos_m[i, 0] - pos_m[j, 0]
                    dy = pos_m[i, 1] - pos_m[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    min_dist = rad_m[i] + rad_m[j]
                    if dist < min_dist:
                        pen_m += penalty_weight * (min_dist - dist)**2
            for i in range(N):
                xi, yi = pos_m[i]
                ri = rad_m[i]
                min_edge_dist = min(xi, 1.0-xi, yi, 1.0-yi)
                if ri > min_edge_dist:
                    pen_m += penalty_weight * (ri - min_edge_dist)**2
            val_m = obj_m + pen_m
            
            grad[k] = (val_p - val_m) / (2*eps)
        
        return total, grad
    
    # Use SLSQP with penalty-based objective
    # We'll do multiple optimization passes with decreasing penalty weight
    
    best_x = x0.copy()
    best_obj = objective_and_grad(x0)[0]
    
    # Optimization passes
    penalty_schedule = [500.0, 200.0, 100.0]
    
    for pw in penalty_schedule:
        if time.time() > deadline - 30:
            break
        
        # Redefine objective with current penalty weight
        def obj_func(x, pw=pw):
            positions = x[:2*N].reshape(N, 2)
            radii = x[2*N:]
            obj = -np.sum(radii)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        penalty += pw * (min_dist - dist)**2
            for i in range(N):
                xi, yi = positions[i]
                ri = radii[i]
                min_edge_dist = min(xi, 1.0-xi, yi, 1.0-yi)
                if ri > min_edge_dist:
                    penalty += pw * (ri - min_edge_dist)**2
            return obj + penalty
        
        # Bounds: positions in [0,1], radii in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x_i
            bounds.append((0.0, 1.0))  # y_i
        for i in range(N):
            bounds.append((0.0, 0.5))  # r_i
        
        result = minimize(obj_func, best_x, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
        
        if result.success and result.fun < best_obj:
            best_x = result.x.copy()
            best_obj = result.fun
    
    # Extract solution
    positions = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    
    # Polish pass: tighten constraints
    if time.time() < deadline - 20:
        # Do a final optimization with very high penalty and tight tolerances
        def obj_func_tight(x):
            positions = x[:2*N].reshape(N, 2)
            radii = x[2*N:]
            obj = -np.sum(radii)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        penalty += 1000.0 * (min_dist - dist)**2
            for i in range(N):
                xi, yi = positions[i]
                ri = radii[i]
                min_edge_dist = min(xi, 1.0-xi, yi, 1.0-yi)
                if ri > min_edge_dist:
                    penalty += 1000.0 * (ri - min_edge_dist)**2
            return obj + penalty
        
        result = minimize(obj_func_tight, best_x, method='SLSQP', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
        
        if result.success:
            best_x = result.x.copy()
            positions = best_x[:2*N].reshape(N, 2)
            radii = best_x[2*N:]
    
    # Final validity check: clamp radii to ensure no overlaps/boundary violations
    # Reduce radii slightly to ensure strict non-overlap
    for i in range(N):
        # Check boundary
        max_r_boundary = min(positions[i, 0], 1.0 - positions[i, 0], 
                             positions[i, 1], 1.0 - positions[i, 1])
        radii[i] = min(radii[i], max_r_boundary - 1e-6)
    
    # Check and fix overlaps
    for _ in range(5):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                required = radii[i] + radii[j]
                if dist < required - 1e-8:
                    # Reduce the larger radius
                    if radii[i] >= radii[j]:
                        radii[i] = max(0.0, (dist - 1e-6) / 2.0)
                    else:
                        radii[j] = max(0.0, (dist - 1e-6) / 2.0)
                    changed = True
        if not changed:
            break
    
    # Ensure all radii are positive
    radii = np.maximum(radii, 1e-4)
    
    # Build result array
    result_arr = np.zeros((N, 3))
    result_arr[:, 0] = positions[:, 0]
    result_arr[:, 1] = positions[:, 1]
    result_arr[:, 2] = radii
    
    return result_arr


def construct():
    """Main entry point with deadline guard."""
    start_time = time.time()
    try:
        result = _construct()
        return result
    except Exception as e:
        # Fallback: return grid solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
