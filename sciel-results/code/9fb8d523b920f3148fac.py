"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    # Objective: maximize sum of radii.
    # We reformulate this as minimizing -sum(r) subject to constraints,
    # or more robustly, using a penalty method that strongly penalizes violations
    # to guide L-BFGS-B towards a valid, dense packing.
    
    # State vector: [x0, y0, r0, x1, y1, r1, ...] length 3*N
    # Constraints:
    # 1. x_i >= r_i, x_i <= 1 - r_i  =>  r_i <= x_i <= 1 - r_i
    # 2. y_i >= r_i, y_i <= 1 - r_i  =>  r_i <= y_i <= 1 - r_i
    # 3. dist(i,j) >= r_i + r_j
    
    # We use a penalty function that is differentiable.
    # Penalty = sum( max(0, violation)^2 )
    # This encourages the optimizer to push circles into contact.

    def objective(state):
        x = state[0::3]
        y = state[1::3]
        r = state[2::3]
        
        # Term 1: Minimize negative sum of radii (to maximize sum)
        obj = -np.sum(r)
        
        # Term 2: Boundary violations
        # Violation if r > x, r > 1-x, r > y, r > 1-y
        v_x = np.maximum(0, r - x)
        v_1x = np.maximum(0, r - (1 - x))
        v_y = np.maximum(0, r - y)
        v_1y = np.maximum(0, r - (1 - y))
        boundary_penalty = np.sum(v_x**2 + v_1x**2 + v_y**2 + v_1y**2)
        
        # Term 3: Overlap violations
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                # Use sqrt for distance, but smooth max is tricky.
                # We can use: max(0, r_i + r_j - dist)^2
                # To avoid sqrt(0) issues, we can clamp dist_sq to small positive.
                dist = np.sqrt(dist_sq + 1e-12)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    overlap_penalty += overlap**2
        
        # Weight the penalties heavily to ensure validity
        weight = 100.0
        return obj + weight * (boundary_penalty + overlap_penalty)

    def gradient(state):
        # Numerical gradient is safer for complex penalty terms, 
        # but analytic is faster. Given 120s budget, numerical might be OK 
        # if we limit iterations, but analytic is better.
        # Let's stick to numerical for robustness against implementation errors in analytic grad,
        # but L-BFGS-B with finite differences can be slow for 78 variables.
        # Let's try analytic gradient for speed.
        
        x = state[0::3].copy()
        y = state[1::3].copy()
        r = state[2::3].copy()
        
        grad = np.zeros(3*N)
        weight = 100.0
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Gradient of boundary penalties
        # v_x = max(0, r - x)
        mask_vx = r > x
        grad[2::3][mask_vx] += weight * 2 * (r[mask_vx] - x[mask_vx])
        grad[0::3][mask_vx] -= weight * 2 * (r[mask_vx] - x[mask_vx])
        
        # v_1x = max(0, r - (1 - x)) = max(0, r - 1 + x)
        mask_v1x = r > (1 - x)
        grad[2::3][mask_v1x] += weight * 2 * (r[mask_v1x] - 1 + x[mask_v1x])
        grad[0::3][mask_v1x] += weight * 2 * (r[mask_v1x] - 1 + x[mask_v1x])
        
        # v_y = max(0, r - y)
        mask_vy = r > y
        grad[2::3][mask_vy] += weight * 2 * (r[mask_vy] - y[mask_vy])
        grad[1::3][mask_vy] -= weight * 2 * (r[mask_vy] - y[mask_vy])
        
        # v_1y = max(0, r - (1 - y))
        mask_v1y = r > (1 - y)
        grad[2::3][mask_v1y] += weight * 2 * (r[mask_v1y] - 1 + y[mask_v1y])
        grad[1::3][mask_v1y] += weight * 2 * (r[mask_v1y] - 1 + y[mask_v1y])
        
        # Gradient of overlap penalties
        # P_ij = (r_i + r_j - dist_ij)^2 if overlap else 0
        # dP/dx_i = 2 * (r_i + r_j - dist) * (-(x_i - x_j)/dist)
        # dP/dx_j = 2 * (r_i + r_j - dist) * ((x_i - x_j)/dist)
        # dP/dr_i = 2 * (r_i + r_j - dist)
        # dP/dr_j = 2 * (r_i + r_j - dist)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    factor = weight * 2 * overlap
                    # d/dx_i
                    grad[3*i]   -= factor * (dx / dist)
                    # d/dx_j
                    grad[3*j]   += factor * (dx / dist)
                    # d/dy_i
                    grad[3*i+1] -= factor * (dy / dist)
                    # d/dy_j
                    grad[3*j+1] += factor * (dy / dist)
                    # d/dr_i
                    grad[3*i+2] += factor
                    # d/dr_j
                    grad[3*j+2] += factor
                    
        return grad

    def is_valid(state, tol=1e-5):
        x = state[0::3]
        y = state[1::3]
        r = state[2::3]
        
        # Bounds
        if np.any(r < -tol) or np.any(x < r - tol) or np.any(x > 1 - r + tol) or \
           np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
           
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def get_score(state):
        x = state[0::3]
        y = state[1::3]
        r = state[2::3]
        # Return sum of radii if valid, else 0 (or penalized value)
        if is_valid(state):
            return np.sum(r)
        return 0.0

    # Initial state: Grid
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    
    count = 0
    for i in range(side):
        for j in range(side):
            if count < N:
                x0[count] = (i + 0.5) / side
                y0[count] = (j + 0.5) / side
                r0[count] = 0.4 / side  # Start slightly smaller to allow growth
                count += 1
    while count < N:
        x0[count] = 0.5
        y0[count] = 0.5
        r0[count] = 0.01
        count += 1
        
    initial_state = np.zeros(3*N)
    initial_state[0::3] = x0
    initial_state[1::3] = y0
    initial_state[2::3] = r0

    best_state = initial_state.copy()
    best_score = get_score(best_state)

    # Bounds for L-BFGS-B
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    # Note: The penalty handles the tighter constraints (x>=r etc),
    # so we can use loose bounds to allow the optimizer to explore.
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Run optimization
    rng = np.random.default_rng(42)
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        if restart == 0:
            x_start = initial_state
        else:
            # Perturb best state or random grid
            if rng.random() < 0.5:
                x_start = best_state + rng.normal(0, 0.05, 3*N)
            else:
                # Random grid
                x_s = np.random.rand(N)
                y_s = np.random.rand(N)
                r_s = np.random.rand(N) * 0.1
                x_start = np.zeros(3*N)
                x_start[0::3] = x_s
                x_start[1::3] = y_s
                x_start[2::3] = r_s
            
        # Ensure start is roughly within [0,1]
        x_start[0::3] = np.clip(x_start[0::3], 0.01, 0.99)
        x_start[1::3] = np.clip(x_start[1::3], 0.01, 0.99)
        x_start[2::3] = np.clip(x_start[2::3], 0.001, 0.49)

        try:
            res = minimize(
                objective, 
                x_start, 
                method='L-BFGS-B', 
                jac=gradient,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            final_state = res.x
            # Ensure no negative radii
            final_state[2::3] = np.maximum(0, final_state[2::3])
            
            score = get_score(final_state)
            if score > best_score:
                best_state = final_state
                best_score = score
                
        except Exception as e:
            pass

    # Final validation and formatting
    x = best_state[0::3]
    y = best_state[1::3]
    r = best_state[2::3]
    
    # If the best state found is invalid (due to numerical noise in penalty), 
    # we might need a final repair step. 
    # However, L-BFGS-B with strong penalties usually converges to valid states.
    # Let's check validity.
    if not is_valid(best_state, tol=1e-4):
        # Simple repair: shrink radii slightly
        r = r * 0.99
        # Clamp to bounds
        for i in range(N):
            max_r = min(x[i], 1-x[i], y[i], 1-y[i])
            r[i] = min(r[i], max_r)
            
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
