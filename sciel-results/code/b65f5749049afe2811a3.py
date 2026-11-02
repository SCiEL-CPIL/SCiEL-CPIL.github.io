"""Circle packing in unit square via parallel tempering Monte Carlo."""
import numpy as np
from scipy.optimize import linprog

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

def _feasible(x, y, r):
    """Check if circles are feasible (disjoint interiors, inside unit square)."""
    n = len(r)
    # Check boundaries
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise distances
    if n < 2:
        return True
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    req_dist = r[:, None] + r[None, :]
    # Mask out diagonal
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    # Check if any pair violates dist < req_dist - epsilon
    if np.any(dist[mask] < req_dist[mask] - 1e-9):
        return False
    return True

def _objective(x, y, r):
    """Sum of radii."""
    return np.sum(r)

def _penalized_objective(x, y, r, beta):
    """Objective with penalty for constraint violations."""
    obj = np.sum(r)
    # Boundary penalties
    obj -= beta * np.sum(np.maximum(0, r - x)**2 + np.maximum(0, r - (1 - x))**2)
    obj -= beta * np.sum(np.maximum(0, r - y)**2 + np.maximum(0, r - (1 - y))**2)
    # Pairwise overlap penalties
    if N < 2:
        return obj
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    req_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0, req_dist - dist)
    # Zero out diagonal
    np.fill_diagonal(overlap, 0)
    obj -= beta * np.sum(overlap**2)
    return obj

def _random_init():
    """Random initial configuration."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.01, 0.08, N)
    return x, y, r

def _move(x, y, r, idx, scale_pos, scale_r):
    """Make a random move on circle idx."""
    x_new = x.copy()
    y_new = y.copy()
    r_new = r.copy()
    
    # Move position
    x_new[idx] += np.random.normal(0, scale_pos)
    y_new[idx] += np.random.normal(0, scale_pos)
    
    # Move radius
    r_new[idx] += np.random.normal(0, scale_r)
    r_new[idx] = np.clip(r_new[idx], 0.005, 0.45)
    
    # Clamp positions to stay in bounds (allow touching)
    x_new[idx] = np.clip(x_new[idx], r_new[idx], 1 - r_new[idx])
    y_new[idx] = np.clip(y_new[idx], r_new[idx], 1 - r_new[idx])
    
    return x_new, y_new, r_new

def _parallel_tempering(num_replicas=10, iterations_per_replica=5000, 
                         temp_base=0.02, temp_factor=1.5,
                         scale_pos=0.02, scale_r=0.005,
                         beta=1000.0, seed=42):
    """Parallel tempering Monte Carlo for circle packing."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize replicas
    states = []
    objectives = []
    for _ in range(num_replicas):
        x, y, r = _random_init()
        states.append((x, y, r))
        objectives.append(_penalized_objective(x, y, r, beta))
    
    # Temperatures (geometric progression)
    temps = np.array([temp_base * (temp_factor ** i) for i in range(num_replicas)])
    
    best_state = None
    best_obj = -np.inf
    
    # Track best feasible solution
    for rep in range(num_replicas):
        x, y, r = states[rep]
        if _feasible(x, y, r):
            obj = _objective(x, y, r)
            if obj > best_obj:
                best_obj = obj
                best_state = (x.copy(), y.copy(), r.copy())
    
    # Main loop
    for it in range(iterations_per_replica):
        # For each replica, make a move
        for rep in range(num_replicas):
            x, y, r = states[rep]
            idx = np.random.randint(0, N)
            x_new, y_new, r_new = _move(x, y, r, idx, scale_pos, scale_r)
            
            obj_old = objectives[rep]
            obj_new = _penalized_objective(x_new, y_new, r_new, beta)
            
            # Metropolis acceptance
            delta_obj = obj_new - obj_old
            if delta_obj >= 0 or np.random.random() < np.exp(delta_obj / temps[rep]):
                states[rep] = (x_new, y_new, r_new)
                objectives[rep] = obj_new
                
                # Check if feasible and update best
                if _feasible(x_new, y_new, r_new):
                    obj_feas = _objective(x_new, y_new, r_new)
                    if obj_feas > best_obj:
                        best_obj = obj_feas
                        best_state = (x_new.copy(), y_new.copy(), r_new.copy())
        
        # Swap adjacent replicas
        for rep in range(num_replicas - 1):
            rep_a = rep
            rep_b = rep + 1
            
            x_a, y_a, r_a = states[rep_a]
            x_b, y_b, r_b = states[rep_b]
            obj_a = objectives[rep_a]
            obj_b = objectives[rep_b]
            T_a = temps[rep_a]
            T_b = temps[rep_b]
            
            # Swap probability
            delta = (obj_b - obj_a) * (1.0 / T_a - 1.0 / T_b)
            if delta >= 0 or np.random.random() < np.exp(delta):
                states[rep_a] = (x_b, y_b, r_b)
                states[rep_b] = (x_a, y_a, r_a)
                objectives[rep_a] = obj_b
                objectives[rep_b] = obj_a
    
    return best_state, best_obj

def _local_optimize(x, y, r, max_iters=2000):
    """Local optimization using clearance-based search."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    current_obj = _objective(x, y, r)
    
    for it in range(max_iters):
        # Calculate clearance for each circle
        # Clearance = min(min_dist_to_neighbor - r_i - r_j, min_dist_to_wall - r_i)
        clearances = np.zeros(N)
        
        for i in range(N):
            # Wall clearance
            c_wall = min(x[i] - r[i], 1 - x[i] - r[i], y[i] - r[i], 1 - y[i] - r[i])
            
            # Neighbor clearance
            if N > 1:
                dists = np.sqrt((x - x[i])**2 + (y - y[i])**2)
                dists[i] = np.inf
                min_dist = np.min(dists)
                # Find the neighbor with min dist to subtract its radius
                j = np.argmin(dists)
                c_neighbor = min_dist - r[i] - r[j]
            else:
                c_neighbor = np.inf
                
            clearances[i] = min(c_wall, c_neighbor)
        
        # Pick the circle with the smallest clearance
        i_min = np.argmin(clearances)
        
        # Try to increase radius of the circle with largest clearance
        i_max = np.argmax(clearances)
        if clearances[i_max] > 1e-6:
            dr = clearances[i_max] * 0.5
            r_test = r.copy()
            r_test[i_max] += dr
            if _feasible(x, y, r_test):
                r = r_test
                current_obj = _objective(x, y, r)
                continue
        
        # Try to move the most constrained circle to increase its clearance
        # Random nudge
        moved = False
        for _ in range(5):
            dx = np.random.normal(0, 0.01)
            dy = np.random.normal(0, 0.01)
            x_new = x.copy()
            y_new = y.copy()
            x_new[i_min] += dx
            y_new[i_min] += dy
            x_new[i_min] = np.clip(x_new[i_min], r[i_min], 1 - r[i_min])
            y_new[i_min] = np.clip(y_new[i_min], r[i_min], 1 - r[i_min])
            
            if _feasible(x_new, y_new, r):
                # Check if clearance improved
                # Recalculate clearance for i_min
                c_wall_new = min(x_new[i_min] - r[i_min], 1 - x_new[i_min] - r[i_min], 
                                 y_new[i_min] - r[i_min], 1 - y_new[i_min] - r[i_min])
                if N > 1:
                    dists_new = np.sqrt((x_new - x_new[i_min])**2 + (y_new - y_new[i_min])**2)
                    dists_new[i_min] = np.inf
                    min_dist_new = np.min(dists_new)
                    j = np.argmin(dists_new)
                    c_neighbor_new = min_dist_new - r[i_min] - r[j]
                else:
                    c_neighbor_new = np.inf
                c_new = min(c_wall_new, c_neighbor_new)
                
                if c_new > clearances[i_min]:
                    x = x_new
                    y = y_new
                    moved = True
                    break
        
        if not moved:
            # If no improvement, try a small random move for any circle
            idx = np.random.randint(N)
            x_new = x.copy()
            y_new = y.copy()
            x_new[idx] += np.random.normal(0, 0.005)
            y_new[idx] += np.random.normal(0, 0.005)
            x_new[idx] = np.clip(x_new[idx], r[idx], 1 - r[idx])
            y_new[idx] = np.clip(y_new[idx], r[idx], 1 - r[idx])
            if _feasible(x_new, y_new, r):
                x = x_new
                y = y_new

    return x, y, r, _objective(x, y, r)

def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    # Run parallel tempering
    best_state, best_obj = _parallel_tempering(
        num_replicas=10,
        iterations_per_replica=5000,
        temp_base=0.02,
        temp_factor=1.5,
        scale_pos=0.02,
        scale_r=0.005,
        beta=1000.0,
        seed=123
    )
    
    if best_state is None:
        # Fallback: try random feasible configs
        for _ in range(1000):
            x, y, r = _random_init()
            if _feasible(x, y, r):
                best_state = (x, y, r)
                break
    
    if best_state is None:
        # Last resort: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r = best_state
    
    # Local optimization
    x, y, r, obj = _local_optimize(x, y, r, max_iters=2000)
    
    # Final feasibility check
    if not _feasible(x, y, r):
        # Try to fix by shrinking radii slightly
        for _ in range(100):
            r = r * 0.999
            if _feasible(x, y, r):
                break
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
