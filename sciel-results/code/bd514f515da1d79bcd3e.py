import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_score = -np.inf
    best_x = None
    
    # Initial guesses for aspect ratios
    # Perimeter 4 means 2(W+H)=4 => W+H=2.
    # Aspect ratio W/H can vary.
    # Uniform grid seed suggests square-ish, but packing might prefer elongated.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5]
    
    # Function to evaluate a configuration
    def evaluate(x):
        # x: [W, H, x_1..x_N, y_1..y_N, r_1..r_N]
        W, H = x[0], x[1]
        coords = x[2:2+2*N].reshape(N, 2)
        radii = x[2+2*N:2+3*N]
        
        # Objective: maximize sum of radii
        score = np.sum(radii)
        
        # Check constraints for validity (for reporting)
        # But for optimization, we use a penalty function
        
        return score, x

    # Penalty function for optimization
    def penalty_func(x):
        W, H = x[0], x[1]
        coords = x[2:2+2*N].reshape(N, 2)
        radii = x[2+2*N:2+3*N]
        
        # 1. Perimeter constraint: W + H <= 2
        p1 = 0
        if W + H > 2:
            p1 += (W + H - 2)**2
        
        # 2. Boundary constraints
        p2 = 0
        for i in range(N):
            xi, yi, ri = coords[i, 0], coords[i, 1], radii[i]
            # x bounds
            if xi < ri: p2 += (ri - xi)**2
            if xi > W - ri: p2 += (xi - (W - ri))**2
            # y bounds
            if yi < ri: p2 += (ri - yi)**2
            if yi > H - ri: p2 += (yi - (H - ri))**2
        
        # 3. Collision constraints
        p3 = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    p3 += (min_dist - dist)**2
        
        return p1 + p2 + p3

    def objective(x):
        # We want to maximize sum(radii) subject to constraints.
        # Using penalty method: Minimize -sum(radii) + lambda * penalty
        W, H = x[0], x[1]
        radii = x[2+2*N:2+3*N]
        # Ensure W, H, radii are positive
        if W <= 0 or H <= 0 or np.any(radii <= 0):
            return 1e10
        score = -np.sum(radii)
        pen = penalty_func(x)
        return score + 1000.0 * pen

    # Bounds for variables
    # W, H in (0, 2)
    # x, y in (0, 2)
    # r in (0, 1)
    bounds = [(1e-6, 2.0), (1e-6, 2.0)]
    for i in range(N):
        bounds.append((1e-6, 2.0)) # x_i
    for i in range(N):
        bounds.append((1e-6, 2.0)) # y_i
    for i in range(N):
        bounds.append((1e-6, 1.0)) # r_i

    # Multi-start optimization
    np.random.seed(42)
    
    num_restarts = 50
    restarts_done = 0
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2.0 * ar / (1 + ar)
        H = 2.0 / (1 + ar)
        
        # Generate initial positions
        # Random scatter or grid
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            restarts_done += 1
            
            # Initial x, y
            xs = np.random.uniform(0.1, W-0.1, N)
            ys = np.random.uniform(0.1, H-0.1, N)
            
            # Initial radii
            # Estimate max radius based on area
            # Area = W*H. 21 circles.
            # Approx radius ~ sqrt(W*H / (21 * pi))
            r_est = np.sqrt(W*H / (N * np.pi)) * 0.8
            rs = np.full(N, r_est)
            
            # Build initial vector
            x0 = np.concatenate(([W, H], xs, ys, rs))
            
            # Optimize
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-9})
                if res.success or res.fun < 1e6:
                    x_opt = res.x
                    W_opt, H_opt = x_opt[0], x_opt[1]
                    coords_opt = x_opt[2:2+2*N].reshape(N, 2)
                    radii_opt = x_opt[2+2*N:2+3*N]
                    
                    # Check if valid
                    pen = penalty_func(x_opt)
                    if pen < 1e-6:
                        score = np.sum(radii_opt)
                        if score > best_score:
                            best_score = score
                            best_x = x_opt
            except Exception:
                pass
                
    if best_x is not None:
        W, H = best_x[0], best_x[1]
        coords = best_x[2:2+2*N].reshape(N, 2)
        radii = best_x[2+2*N:2+3*N]
        
        # Construct output array
        result = np.zeros((N, 3))
        result[:, 0] = coords[:, 0]
        result[:, 1] = coords[:, 1]
        result[:, 2] = radii
        
        return result
    else:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
