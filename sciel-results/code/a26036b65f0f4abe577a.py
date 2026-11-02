"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # 1. Initial Configuration: Hexagonal Grid
    # We want to pack 26 circles. A hexagonal arrangement is efficient.
    # Let's try to fit a hex grid into the unit square.
    # Estimate radius for hex grid.
    # Density of hex packing is ~0.9069.
    # Area of square is 1. Area of circles ~ 0.9069.
    # N * pi * r^2 ~ 0.9069 => r ~ sqrt(0.9069 / (26 * pi)) ~ sqrt(0.9069/81.68) ~ sqrt(0.0111) ~ 0.105
    # Let's start with a slightly smaller radius to ensure feasibility, e.g., 0.08.
    
    def hex_grid_initial(n):
        # Create a hex grid of points
        # r is the radius of the circles.
        # Distance between centers in hex grid is 2*r.
        # Let's pick a base radius r0.
        r0 = 0.08
        dx = 2 * r0
        dy = 2 * r0 * np.sqrt(3) / 2
        
        points = []
        row = 0
        while len(points) < n:
            # Offset every other row
            offset = dx / 2 if row % 2 else 0
            col = 0
            while len(points) < n:
                x = offset + col * dx
                y = row * dy
                # Check if within bounds [0,1]
                if x + r0 <= 1 and y + r0 <= 1:
                    points.append([x, y, r0])
                col += 1
            row += 1
        
        # If we have more than n, take first n. If less, we need a different strategy.
        # For N=26, r=0.08 should generate plenty.
        # 1/0.16 = 6.25 cols. 1/(0.138) = 7.2 rows. 6*7 = 42 points. Enough.
        
        return np.array(points[:n])

    initial_config = hex_grid_initial(N)
    
    # Ensure strict feasibility initially
    # Check overlaps and boundaries
    def is_valid(config):
        # config: (N, 3) array of [x, y, r]
        # Boundary
        if np.any(config[:, 0] - config[:, 2] < -1e-9) or np.any(1 - config[:, 0] - config[:, 2] < -1e-9):
            return False
        if np.any(config[:, 1] - config[:, 2] < -1e-9) or np.any(1 - config[:, 1] - config[:, 2] < -1e-9):
            return False
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = config[i, 0] - config[j, 0]
                dy = config[i, 1] - config[j, 1]
                dist_sq = dx*dx + dy*dy
                rad_sum = config[i, 2] + config[j, 2]
                if dist_sq < rad_sum*rad_sum - 1e-9:
                    return False
        return True

    if not is_valid(initial_config):
        # Shrink radii until valid
        scale = 1.0
        while not is_valid(initial_config) and scale > 0.1:
            scale *= 0.95
            initial_config[:, 2] *= scale
            # Re-center if needed? For now, just shrink.
            
    best_config = initial_config.copy()
    best_score = np.sum(best_config[:, 2])

    # 2. Optimization using SLSQP
    # Flatten the configuration: [x1, y1, r1, x2, y2, r2, ...]
    x0 = initial_config.flatten()
    
    # Objective: maximize sum of radii => minimize -sum(r)
    def objective(v):
        config = v.reshape(N, 3)
        return -np.sum(config[:, 2])
        
    # Constraints
    # SLSQP expects constraints in the form g(x) >= 0
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    # r_i >= 0 (handled by bounds)
    
    def boundary_constraints(v):
        config = v.reshape(N, 3)
        cons = []
        for i in range(N):
            cons.append(config[i, 0] - config[i, 2])
            cons.append(1 - config[i, 0] - config[i, 2])
            cons.append(config[i, 1] - config[i, 2])
            cons.append(1 - config[i, 1] - config[i, 2])
        return cons
    
    # Pairwise distance constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def distance_constraints(v):
        config = v.reshape(N, 3)
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = config[i, 0] - config[j, 0]
                dy = config[i, 1] - config[j, 1]
                dist_sq = dx*dx + dy*dy
                rad_sum = config[i, 2] + config[j, 2]
                cons.append(dist_sq - rad_sum*rad_sum)
        return cons

    # Combine constraints into a single function for efficiency?
    # SLSQP can take a list of dicts or a single function returning an array.
    # A single function returning an array is often faster.
    
    def all_constraints(v):
        config = v.reshape(N, 3)
        cons = []
        # Boundaries
        for i in range(N):
            cons.append(config[i, 0] - config[i, 2])
            cons.append(1 - config[i, 0] - config[i, 2])
            cons.append(config[i, 1] - config[i, 2])
            cons.append(1 - config[i, 1] - config[i, 2])
        # Distances
        for i in range(N):
            for j in range(i+1, N):
                dx = config[i, 0] - config[j, 0]
                dy = config[i, 1] - config[j, 1]
                dist_sq = dx*dx + dy*dy
                rad_sum = config[i, 2] + config[j, 2]
                cons.append(dist_sq - rad_sum*rad_sum)
        return np.array(cons)

    # Bounds for each variable
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    constraints = {'type': 'ineq', 'fun': all_constraints}

    # Run optimization
    try:
        # Use multiple starts if time permits? 
        # For now, one robust run.
        # maxiter limit to avoid hanging
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=constraints, 
                          options={'maxiter': 500, 'ftol': 1e-12})
        
        if result.success or (time.time() - start_time < deadline):
            config_opt = result.x.reshape(N, 3)
            # Clean up small numerical violations
            # Clip radii to be positive
            config_opt[:, 2] = np.clip(config_opt[:, 2], 1e-6, 0.5)
            
            # Check validity
            if is_valid(config_opt):
                score_opt = np.sum(config_opt[:, 2])
                if score_opt > best_score:
                    best_config = config_opt
                    best_score = score_opt
            else:
                # If slightly invalid, try to repair by shrinking radii
                # or just keep the previous best
                pass
                
    except Exception as e:
        # Fallback to initial config if optimization crashes
        pass

    # If we have time, try a few random restarts or perturbations
    # Given the 150s limit and SLSQP speed, we might have time for 1-2 more tries
    # if the first one was fast.
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try one more optimization with a slightly perturbed start
        # or a different initial radius
        x0_perturbed = x0 + np.random.normal(0, 0.01, size=x0.shape)
        # Ensure perturbed start is valid
        config_pert = x0_perturbed.reshape(N, 3)
        config_pert[:, 2] = np.clip(config_pert[:, 2], 1e-6, 0.5)
        # Simple validity check and repair
        if is_valid(config_pert):
            try:
                result2 = minimize(objective, x0_perturbed, method='SLSQP', bounds=bounds, 
                                   constraints=constraints, 
                                   options={'maxiter': 300, 'ftol': 1e-12})
                if result2.success:
                    config_opt2 = result2.x.reshape(N, 3)
                    config_opt2[:, 2] = np.clip(config_opt2[:, 2], 1e-6, 0.5)
                    if is_valid(config_opt2):
                        score_opt2 = np.sum(config_opt2[:, 2])
                        if score_opt2 > best_score:
                            best_config = config_opt2
                            best_score = score_opt2
            except:
                pass

    return best_config
# EVOLVE-BLOCK-END
