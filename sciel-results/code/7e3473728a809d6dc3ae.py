"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _objective(x, n_circles):
    """
    x: flat array of length 3*n_circles
    Returns the sum of radii (to maximize, we minimize negative).
    Constraints are handled via penalty or by ensuring validity in construct.
    """
    circles = x.reshape(n_circles, 3)
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Rectangle dimensions W, H with W + H = 2 (perimeter 4)
    # We need to determine W and H from the circles? 
    # No, the problem says "variable rectangle of perimeter 4".
    # The bounding box of the circles defines the rectangle? 
    # "n disjoint circles whose bounding rectangle has perimeter at most 4"
    # So the rectangle is the axis-aligned bounding box of the circles.
    # Let x_min = min(cx - r), x_max = max(cx + r)
    # Let y_min = min(cy - r), y_max = max(cy + r)
    # W = x_max - x_min, H = y_max - y_min
    # Perimeter = 2*(W+H) <= 4 => W+H <= 2.
    
    # However, we can also fix the rectangle and place circles inside.
    # If we fix W and H such that W+H=2, we just need circles inside [0,W]x[0,H].
    # Maximizing sum of radii.
    
    # Strategy: Optimize over W, and circle positions/radii.
    # Total variables: 1 (W) + 3*N (circles)
    
    # But the current function signature in the seed was just construct().
    # The prompt says: "Keep the function names and signatures of the seed program."
    # Seed: def construct(): ...
    
    # So I will implement a solver inside construct() or helper functions.
    
    pass

def construct():
    start_time = time.time()
    deadline = 110.0  # 10 seconds safety margin
    
    # Parameters
    n = N
    
    # We optimize over:
    # W (width of rectangle), H = 2 - W
    # For each circle i: x_i, y_i, r_i
    # Constraints:
    # 1. r_i > 0
    # 2. x_i - r_i >= 0, x_i + r_i <= W
    # 3. y_i - r_i >= 0, y_i + r_i <= H
    # 4. dist(i, j) >= r_i + r_j for all i != j
    
    # We can use a penalty method with scipy.optimize.minimize or differential_evolution.
    # Differential evolution is robust for global optimization with bounds.
    
    # Bounds:
    # W in [0.5, 1.5] (symmetric around 1.0)
    # For each circle:
    #   r in [0, 1.0] (max possible radius is limited by W/2 and H/2)
    #   x in [0, W] -> depends on W. This is tricky for DE if bounds depend on another var.
    
    # Alternative: Normalize coordinates.
    # Let rectangle be [0, W] x [0, H].
    # We can define relative positions u_i in [0, 1], v_i in [0, 1].
    # x_i = u_i * W, y_i = v_i * H.
    # Constraints:
    # u_i * W - r_i >= 0  => r_i <= u_i * W
    # u_i * W + r_i <= W => r_i <= (1 - u_i) * W
    # v_i * H - r_i >= 0  => r_i <= v_i * H
    # v_i * H + r_i <= H => r_i <= (1 - v_i) * H
    
    # So r_i <= min(u_i, 1-u_i)*W and r_i <= min(v_i, 1-v_i)*H.
    
    # Objective: Maximize sum(r_i).
    
    # Variables:
    # W (1)
    # For each i: u_i, v_i, r_i (3*N)
    # Total: 3*N + 1 = 64 variables.
    
    # Bounds:
    # W: [0.5, 1.5]
    # u_i, v_i: [0.01, 0.99] (keep away from edges to allow r>0)
    # r_i: [0.0, 0.5]
    
    # Penalty function:
    # For each circle i:
    #   max_r_i = min(min(u_i, 1-u_i)*W, min(v_i, 1-v_i)*H)
    #   if r_i > max_r_i: penalty += (r_i - max_r_i)^2
    # For each pair (i, j):
    #   dx = (u_i - u_j)*W
    #   dy = (v_i - v_j)*H
    #   dist = sqrt(dx^2 + dy^2)
    #   min_dist = r_i + r_j
    #   if dist < min_dist: penalty += (min_dist - dist)^2
    
    # Minimize: -sum(r_i) + lambda * penalty
    
    def objective(params):
        W = params[0]
        H = 2.0 - W
        circles = params[1:].reshape(n, 3)
        u = circles[:, 0]
        v = circles[:, 1]
        r = circles[:, 2]
        
        # Check bounds for W
        if W < 0.1 or H < 0.1:
            return 1e10
        
        sum_r = np.sum(r)
        penalty = 0.0
        
        # Boundary constraints
        # r_i <= min(u_i, 1-u_i)*W
        max_r_x = np.minimum(u, 1.0 - u) * W
        # r_i <= min(v_i, 1-v_i)*H
        max_r_y = np.minimum(v, 1.0 - v) * H
        max_r = np.minimum(max_r_x, max_r_y)
        
        violations = np.maximum(0.0, r - max_r)
        penalty += np.sum(violations**2)
        
        # Pairwise constraints
        # Use vectorized distance calculation if possible, but n=21 is small.
        for i in range(n):
            for j in range(i + 1, n):
                dx = (u[i] - u[j]) * W
                dy = (v[i] - v[j]) * H
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # Scale penalty
        return -sum_r + 100.0 * penalty
    
    # Bounds for DE
    bounds = [(0.5, 1.5)]  # W
    for i in range(n):
        bounds.append((0.05, 0.95))  # u_i
        bounds.append((0.05, 0.95))  # v_i
        bounds.append((0.0, 0.5))    # r_i
    
    # Initial guess: grid
    side = int(np.ceil(np.sqrt(n)))
    init_W = 1.0
    init_params = [init_W]
    for i in range(n):
        ix = i % side
        iy = i // side
        u = (ix + 0.5) / side
        v = (iy + 0.5) / side
        r = 0.5 / side - 0.01
        init_params.extend([u, v, r])
    
    # Run DE
    try:
        # maxiter depends on time
        # Start with a moderate number, increase if time permits
        maxiter = 200
        tol = 1e-4
        
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=maxiter, 
            tol=tol,
            polish=True,
            popsize=15
        )
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Return best found so far? DE doesn't expose intermediate easily.
            # We'll just use the result if it finished, else fallback.
            pass
            
        if result.success or result.fun < 0:
            best_params = result.x
            W = best_params[0]
            H = 2.0 - W
            circles = best_params[1:].reshape(n, 3)
            u, v, r = circles[:, 0], circles[:, 1], circles[:, 2]
            
            # Convert to absolute coords
            cx = u * W
            cy = v * H
            
            # Verify validity
            valid = True
            for i in range(n):
                if cx[i] - r[i] < -1e-6 or cx[i] + r[i] > W + 1e-6:
                    valid = False
                if cy[i] - r[i] < -1e-6 or cy[i] + r[i] > H + 1e-6:
                    valid = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-6:
                        valid = False
            
            if valid:
                res = np.column_stack((cx, cy, r))
                return res
    
    except Exception as e:
        pass
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(n)))
    W = 1.0
    H = 1.0
    cells = []
    for i in range(n):
        ix = i % side
        iy = i // side
        cx = (ix + 0.5) / side * W
        cy = (iy + 0.5) / side * H
        r = 0.5 / side * W - 0.01
        r = min(r, 0.5 / side * H - 0.01)
        cells.append((cx, cy, r))
    return np.array(cells)

# EVOLVE-BLOCK-END
