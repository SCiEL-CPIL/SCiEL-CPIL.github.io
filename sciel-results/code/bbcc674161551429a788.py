import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def construct():
    """
    Pack 26 circles in the unit square maximizing the sum of radii.
    Uses Simulated Annealing with a penalty-based cost function.
    """
    # 1. Initialization: Grid layout to ensure a valid starting point
    side = int(np.ceil(np.sqrt(N)))
    # Generate grid coordinates
    grid_x = []
    grid_y = []
    for i in range(side):
        for j in range(side):
            grid_x.append((i + 0.5) / side)
            grid_y.append((j + 0.5) / side)
            
    # Take the first N
    x = np.array(grid_x[:N], dtype=float)
    y = np.array(grid_y[:N], dtype=float)
    
    # Initial radii: slightly smaller than half the grid spacing to avoid overlap
    # Grid spacing is 1/side. Max radius in cell is 0.5/side.
    # We use 0.4/side to be safe.
    r = np.full(N, 0.4 / side, dtype=float)
    
    # Precompute indices for pair-wise distance calculations
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def calculate_cost(x, y, r):
        """
        Calculate the cost function.
        We minimize Cost = -Sum(r) + Penalty(Overlaps) + Penalty(Boundaries)
        """
        # 1. Boundary Penalties
        # Circle i is inside if: r_i <= x_i <= 1-r_i and r_i <= y_i <= 1-r_i
        # Violation in x: max(0, r_i - x_i) + max(0, r_i - (1 - x_i))
        vx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
        vy = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        boundary_penalty = np.sum(vx**2) + np.sum(vy**2)
        
        # 2. Overlap Penalties
        # Distance between i and j
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Required distance
        req_dist = r[i_idx] + r[j_idx]
        
        # Overlap amount
        overlaps = np.maximum(0, req_dist - dist)
        overlap_penalty = np.sum(overlaps**2)
        
        # 3. Objective: Maximize sum of radii -> Minimize -sum(r)
        sum_r = np.sum(r)
        
        # Weight for penalties
        # High weight ensures feasibility is prioritized
        penalty_weight = 1000.0
        
        return -sum_r + penalty_weight * (overlap_penalty + boundary_penalty)

    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    T = 1.0
    T_min = 0.0001
    cooling_rate = 0.9995
    
    current_cost = calculate_cost(x, y, r)
    
    # Main SA Loop
    while time.time() - start_time < time_limit and T > T_min:
        # Select a random circle to perturb
        k = np.random.randint(N)
        
        # Save old state
        old_xk = x[k]
        old_yk = y[k]
        old_rk = r[k]
        
        # Perturbation magnitude scales with temperature
        sigma = 0.02 * T
        
        # Apply perturbations
        x[k] += np.random.normal(0, sigma)
        y[k] += np.random.normal(0, sigma)
        r[k] += np.random.normal(0, sigma)
        
        # Ensure radius is positive
        if r[k] < 1e-4:
            r[k] = 1e-4
            
        # Calculate new cost
        new_cost = calculate_cost(x, y, r)
        
        # Metropolis Acceptance Criterion
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            current_cost = new_cost
        else:
            # Revert changes
            x[k] = old_xk
            y[k] = old_yk
            r[k] = old_rk
            
        # Cool down
        T *= cooling_rate
        
    # Final Validation and Repair
    # The SA might leave very small overlaps or boundary violations due to floating point 
    # or if the penalty weight wasn't high enough in the late stages.
    # We perform a simple repair to ensure strict validity.
    
    # 1. Repair Boundaries
    for i in range(N):
        # Max radius allowed by boundaries for current x, y
        max_r_x = min(x[i], 1.0 - x[i])
        max_r_y = min(y[i], 1.0 - y[i])
        max_r = min(max_r_x, max_r_y)
        
        if r[i] > max_r:
            r[i] = max_r * 0.99 # Shrink slightly to be safe
            
    # 2. Repair Overlaps (Iterative shrinkage)
    # This is a conservative repair. If overlaps exist, shrink the larger circle or both.
    # We do a few iterations to resolve conflicts.
    for _ in range(10):
        adjusted = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Overlap exists
                    # Shrink radii to satisfy constraint
                    # We want to keep sum(r) as high as possible.
                    # Simple heuristic: shrink both proportionally to their radii
                    # such that r_i' + r_j' = dist * 0.99
                    
                    target_sum = dist * 0.99
                    current_sum = r[i] + r[j]
                    
                    if current_sum > 0:
                        scale = target_sum / current_sum
                        if scale < 1.0:
                            r[i] *= scale
                            r[j] *= scale
                            adjusted = True
        if not adjusted:
            break
            
    # Final check: ensure no negative radii
    r = np.maximum(r, 0.0)
    
    return np.column_stack((x, y, r))

# EVOLVE-BLOCK-END
