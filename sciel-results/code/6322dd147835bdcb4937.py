import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_val = -np.inf
    best_solution = None

    # Objective function with penalty
    def objective(params):
        # params: x[0:N], y[0:N], r[0:N], w[N*3]
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        # Ensure w, h positive
        if w <= 0.1 or h <= 0.1:
            return 1e6

        penalty = 0.0
        
        # Wall constraints
        for i in range(N):
            min_dist_wall = min(x[i], w - x[i], y[i], h - y[i])
            if r[i] > min_dist_wall:
                penalty += (r[i] - min_dist_wall) ** 2
            # Also ensure r[i] is not negative
            if r[i] < 0:
                penalty += r[i] ** 2
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        sum_r = np.sum(r)
        
        # Penalty weight
        # Start high to enforce constraints, could anneal
        mu = 1000.0 
        
        return -sum_r + mu * penalty

    # Initial guess: 5x5 grid on 1x1 square
    # We use 21 circles. A 5x5 grid has 25. We take the first 21.
    # To be more symmetric, we can center the 21.
    # Let's just use a standard grid and let the optimizer move them.
    
    side = 5
    grid_coords = []
    for i in range(side):
        for j in range(side):
            grid_coords.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # Take 21
    init_coords = grid_coords[:N]
    init_x = np.array([c[0] for c in init_coords])
    init_y = np.array([c[1] for c in init_coords])
    init_r = np.full(N, 0.1) # 0.5 / 5
    init_w = 1.0
    
    x0 = np.concatenate([init_x, init_y, init_r, [init_w]])
    
    # Run Nelder-Mead
    try:
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={
                'maxiter': 10000, 
                'xatol': 1e-8, 
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        # Check time
        if time.time() > deadline:
            raise TimeoutError()

        if res.fun < best_val:
            # Extract solution
            x = res.x[0:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            w = res.x[3*N]
            h = 2.0 - w
            
            # Validate strictly
            valid = True
            if w < 0.1 or h < 0.1:
                valid = False
            else:
                for i in range(N):
                    if r[i] < -1e-9:
                        valid = False
                        break
                    min_dist_wall = min(x[i], w - x[i], y[i], h - y[i])
                    if r[i] > min_dist_wall + 1e-9:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                            if dist < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                # Clip radii to be non-negative and within bounds for safety
                r = np.maximum(r, 0)
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_solution = np.column_stack((x, y, r))
                    
    except Exception as e:
        pass

    # If we have time, try a second start with a different aspect ratio
    if time.time() < deadline - 20 and best_solution is not None:
        # Try w = 1.2, h = 0.8
        # Rescale coordinates? No, just change init_w and let it adjust.
        # Actually, if we change w, the initial x,y might be out of bounds.
        # Let's just stick to the best found so far if the first one is good.
        # The 1x1 square is a very strong candidate for N=21.
        pass

    if best_solution is None:
        # Fallback: Return the initial grid if optimization failed
        # Ensure it's valid
        return np.column_stack((init_x, init_y, init_r))
        
    return best_solution

# EVOLVE-BLOCK-END
