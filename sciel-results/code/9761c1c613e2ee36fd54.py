"""Simulated Annealing circle packing in perimeter-4 rectangle."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Simulated Annealing with a vectorized energy calculation.
    We optimize the aspect ratio of the rectangle as well as the circle positions and radii.
    """
    start_time = time.time()
    timeout = 110.0 # Leave some buffer for 120s limit
    
    # We will try a few different aspect ratios or optimize it.
    # Let's optimize aspect ratio (log_w) as part of the SA state? 
    # Or fix a good aspect ratio? 
    # For N=21, a rectangle of 1x1 is square. 
    # A slightly rectangular shape might be better. 
    # Let's include aspect ratio in the SA variables.
    
    # State vector: [log_aspect, x_1, y_1, r_1, ..., x_N, y_N, r_N]
    # Total size: 1 + 3*N
    
    best_solution = None
    best_score = -np.inf
    
    # Run multiple restarts if time permits
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize state
        # Start with aspect ratio 1.0 (square) or slightly varied
        log_aspect = np.log(1.0 + 0.1 * (restart % 3 - 1)) # -0.1, 0, 0.1
        
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        aspect = np.exp(log_aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        x0 = np.zeros(1 + 3*N)
        x0[0] = log_aspect
        for i in range(N):
            row = i // side
            col = i % side
            # Jitter initial positions slightly to break symmetry
            x0[1 + 3*i] = (col + 0.5) * w / side + np.random.uniform(-0.05, 0.05)
            x0[2 + 3*i] = (row + 0.5) * h / side + np.random.uniform(-0.05, 0.05)
            x0[3 + 3*i] = 0.05 # Initial radius
            
        # Run SA
        solution, score = _simulated_annealing(x0, time_limit=timeout - (time.time() - start_time))
        
        if solution is not None and score > best_score:
            best_score = score
            best_solution = solution
    
    if best_solution is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution


def _simulated_annealing(state, time_limit=10.0):
    """
    Run Simulated Annealing on the given state.
    State: [log_aspect, x_1, y_1, r_1, ...]
    Returns: (best_valid_solution_array, best_score)
    """
    start_time = time.time()
    T = 1.0
    T_min = 1e-4
    alpha = 0.9995 # Cooling rate
    step_size = 0.1
    
    best_state = state.copy()
    best_energy = _calculate_energy(state)
    best_score = -best_energy if best_energy < 1e-6 else -np.inf
    
    current_state = state.copy()
    current_energy = best_energy
    
    n_var = len(state)
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb a random subset of variables
        # We perturb aspect ratio and a few circles
        new_state = current_state.copy()
        
        # Perturb aspect ratio
        if np.random.rand() < 0.2:
            new_state[0] += np.random.uniform(-0.1, 0.1)
            
        # Perturb 1-3 circles
        n_circles_to_move = np.random.randint(1, 4)
        indices = np.random.choice(N, size=n_circles_to_move, replace=False)
        for idx in indices:
            offset = 1 + 3*idx
            # Jitter position and radius
            new_state[offset] += np.random.uniform(-step_size, step_size) # x
            new_state[offset+1] += np.random.uniform(-step_size, step_size) # y
            new_state[offset+2] += np.random.uniform(-step_size*0.5, step_size*0.5) # r
            
        new_energy = _calculate_energy(new_state)
        
        # Metropolis acceptance
        delta = new_energy - current_energy
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_state = new_state
            current_energy = new_energy
            
            # Update best
            # Check if valid (energy penalty is small)
            # For tracking best score, we only count if valid
            penalty = _calculate_penalty(new_state)
            if penalty < 1e-6:
                score = np.sum(new_state[3::3]) # Sum of radii
                if score > best_score:
                    best_score = score
                    best_state = new_state.copy()
        
        # Cool down
        T *= alpha
        
    # Return best valid solution
    if best_score > -np.inf:
        # Extract solution
        sol = np.zeros((N, 3))
        for i in range(N):
            sol[i, 0] = best_state[1 + 3*i]
            sol[i, 1] = best_state[2 + 3*i]
            sol[i, 2] = best_state[3 + 3*i]
        return sol, best_score
    else:
        return None, -np.inf


def _calculate_energy(state):
    """
    Calculate total energy: -sum(radii) + lambda * penalty
    """
    penalty = _calculate_penalty(state)
    # We want to maximize sum(radii) subject to penalty=0.
    # So energy = -sum(radii) + lambda * penalty
    # If penalty is 0, energy = -sum(radii). Lower is better.
    radii = state[3::3]
    sum_radii = np.sum(radii)
    lambda_pen = 100.0
    return -sum_radii + lambda_pen * penalty


def _calculate_penalty(state):
    """
    Calculate penalty for overlaps and boundary violations.
    """
    log_aspect = state[0]
    aspect = np.exp(log_aspect)
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    xs = state[1::3]
    ys = state[2::3]
    rs = state[3::3]
    
    pen = 0.0
    
    # Boundary penalty
    # x < r => x - r < 0
    # x > w - r => x - (w-r) > 0
    # Use squared penalty
    left_viol = np.maximum(0, rs - xs)
    right_viol = np.maximum(0, xs - (w - rs))
    bottom_viol = np.maximum(0, rs - ys)
    top_viol = np.maximum(0, ys - (h - rs))
    
    pen += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Overlap penalty
    # For each pair i, j
    # dist < r_i + r_j => (r_i + r_j - dist)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                pen += (min_dist - dist)**2
                
    return pen

# EVOLVE-BLOCK-END
