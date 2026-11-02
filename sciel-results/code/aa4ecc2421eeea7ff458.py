import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_result = None
    
    # Parameters for Simulated Annealing
    T_start = 0.05
    T_end = 1e-5
    cooling_rate = 0.9998
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        np.random.seed(restart * 12345 + 1)
        
        # Initialize with random positions, slightly inset to avoid immediate boundary issues
        centers = np.random.uniform(0.05, 0.95, size=(N, 2))
        radii = np.zeros(N)
        
        # Initial radii: set to max possible given current positions (greedy init)
        # This ensures we start with a valid configuration
        for i in range(N):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            # Max radius is limited by distance to others and boundaries
            # r_i <= (d_ij - r_j) / 2
            # Since we start with r_j=0, r_i <= d_ij / 2
            # Also r_i <= min(x, 1-x, y, 1-y)
            candidate_r = np.min((dists[mask]) / 2.0)
            bound_r = min(centers[i][0], 1.0 - centers[i][0], centers[i][1], 1.0 - centers[i][1])
            radii[i] = min(candidate_r, bound_r, 0.1) # Cap initial radius to prevent huge overlaps in init
            
        T = T_start
        steps = 0
        max_steps = 30000
        
        current_energy = -np.sum(radii)
        
        while T > T_end and steps < max_steps:
            if time.time() > deadline:
                break
                
            # Pick a random circle
            i = np.random.randint(N)
            
            # Generate a new random position
            # Mix of random and local perturbation
            if np.random.rand() < 0.3:
                new_center = np.random.uniform(0.0, 1.0, size=2)
            else:
                step_size = 0.05 * (T / T_start) + 0.005
                new_center = centers[i] + np.random.normal(0, step_size, size=2)
                new_center = np.clip(new_center, 0.0, 1.0)
            
            # Calculate new radius for circle i at new_center
            dists = np.linalg.norm(centers - new_center, axis=1)
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            
            # Constraint from other circles: d_ij >= r_i + r_j => r_i <= d_ij - r_j
            # Note: We must ensure d_ij > r_j for valid r_i > 0
            # If d_ij <= r_j, then r_i can be 0 (or negative, which we clamp to 0)
            r_constrained_by_others = np.min(dists[mask] - radii[mask])
            
            # Constraint from boundaries
            r_constrained_by_bounds = min(new_center[0], 1.0 - new_center[0], new_center[1], 1.0 - new_center[1])
            
            new_radius = max(0.0, min(r_constrained_by_others, r_constrained_by_bounds))
            
            # Calculate energy change
            # Energy = -sum(radii)
            # Old contribution: -radii[i]
            # New contribution: -new_radius
            delta_energy = -(new_radius - radii[i])
            
            # Metropolis criterion
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
                # Accept move
                centers[i] = new_center
                radii[i] = new_radius
                current_energy += delta_energy
                
                # Track best
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((centers, radii)).copy()
            
            # Cool down
            T *= cooling_rate
            steps += 1
            
        # Refine the best configuration found in this restart using gradient-based optimization
        # This combines the global search of SA with the local refinement of L-BFGS-B
        
        if best_result is not None:
            # Use the best result from this restart as the initial guess for refinement
            # Note: best_result is global, but we want to refine the current local best
            # Let's refine the current state (centers, radii) if it's good, or the global best
            # Actually, let's refine the global best found so far, as it's the most promising.
            # But wait, SA might have found a local optimum that is better than the global best tracked?
            # No, best_score tracks the global best. So we refine best_result.
            
            # Prepare params for scipy
            p0 = np.concatenate([best_result[:, 0], best_result[:, 1], best_result[:, 2]])
            
            # Objective function: minimize -sum(r) + penalty * (violations)
            # We use a high penalty to enforce constraints strictly during refinement.
            penalty = 1000.0
            
            def objective(v):
                if time.time() > deadline:
                    return 1e10
                x = v[0::3]
                y = v[1::3]
                r = v[2::3]
                
                sum_r = np.sum(r)
                penalty_val = 0.0
                
                # Boundary constraints
                for i in range(N):
                    if x[i] < r[i]:
                        penalty_val += (r[i] - x[i])**2
                    if x[i] > 1.0 - r[i]:
                        penalty_val += (x[i] + r[i] - 1.0)**2
                    if y[i] < r[i]:
                        penalty_val += (r[i] - y[i])**2
                    if y[i] > 1.0 - r[i]:
                        penalty_val += (y[i] + r[i] - 1.0)**2
                
                # Collision constraints
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            penalty_val += (min_dist - dist)**2
                            
                return -sum_r + penalty * penalty_val

            # Gradient
            def gradient(v):
                x = v[0::3]
                y = v[1::3]
                r = v[2::3]
                grad = np.zeros_like(v)
                
                # d(-sum r)/dr = -1
                grad[2::3] = -1.0
                
                # Boundary gradients
                for i in range(N):
                    if x[i] < r[i]:
                        grad[3*i] += 2 * penalty * (r[i] - x[i]) * (-1)
                        grad[3*i+2] += 2 * penalty * (r[i] - x[i]) * (1)
                    if x[i] > 1.0 - r[i]:
                        grad[3*i] += 2 * penalty * (x[i] + r[i] - 1.0) * (1)
                        grad[3*i+2] += 2 * penalty * (x[i] + r[i] - 1.0) * (1)
                    if y[i] < r[i]:
                        grad[3*i+1] += 2 * penalty * (r[i] - y[i]) * (-1)
                        grad[3*i+2] += 2 * penalty * (r[i] - y[i]) * (1)
                    if y[i] > 1.0 - r[i]:
                        grad[3*i+1] += 2 * penalty * (y[i] + r[i] - 1.0) * (1)
                        grad[3*i+2] += 2 * penalty * (y[i] + r[i] - 1.0) * (1)
                
                # Collision gradients
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r[i] + r[j]
                        if dist < min_dist and dist > 1e-9:
                            overlap = min_dist - dist
                            grad[3*i] += 2 * penalty * overlap * (-dx / dist)
                            grad[3*i+1] += 2 * penalty * overlap * (-dy / dist)
                            grad[3*i+2] += 2 * penalty * overlap * (1)
                            
                            grad[3*j] += 2 * penalty * overlap * (dx / dist)
                            grad[3*j+1] += 2 * penalty * overlap * (dy / dist)
                            grad[3*j+2] += 2 * penalty * overlap * (1)
                            
                return grad

            # Run L-BFGS-B
            try:
                res = minimize(objective, p0, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 1000})
                
                if res.success or res.fun < objective(p0):
                    p_opt = res.x
                    x_opt = p_opt[0::3]
                    y_opt = p_opt[1::3]
                    r_opt = p_opt[2::3]
                    
                    # Verify validity of the refined solution
                    valid = True
                    for i in range(N):
                        if x_opt[i] < r_opt[i] - 1e-6 or x_opt[i] > 1.0 - r_opt[i] + 1e-6:
                            valid = False
                            break
                        if y_opt[i] < r_opt[i] - 1e-6 or y_opt[i] > 1.0 - r_opt[i] + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = x_opt[i] - x_opt[j]
                                dy = y_opt[i] - y_opt[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < r_opt[i] + r_opt[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                                
                    if valid:
                        score = np.sum(r_opt)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack((x_opt, y_opt, r_opt)).copy()
            except Exception:
                pass

    # Fallback if no valid result was found
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append((cx, cy, r))
        return np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
