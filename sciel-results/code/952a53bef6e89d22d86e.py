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
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def _objective(x, y, r):
    """Sum of radii."""
    return np.sum(r)

def _penalized_objective(x, y, r, beta):
    """Objective with penalty for constraint violations. Vectorized."""
    obj = np.sum(r)
    
    # Boundary penalties (vectorized)
    # left: x - r < 0 => r - x > 0
    # right: x + r > 1 => x + r - 1 > 0
    # bottom: y - r < 0 => r - y > 0
    # top: y + r > 1 => y + r - 1 > 0
    viol_left = np.maximum(0, r - x)
    viol_right = np.maximum(0, x + r - 1)
    viol_bottom = np.maximum(0, r - y)
    viol_top = np.maximum(0, y + r - 1)
    
    boundary_violation = np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)
    
    # Pairwise overlap penalties (vectorized)
    # Compute distance matrix
    # D[i, j] = sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2)
    # We only need upper triangle for i < j
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx**2 + dy**2)
    
    # Required separation: r[i] + r[j]
    sep_mat = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Overlap: max(0, sep - dist)
    # We only care about i < j, so we can sum over upper triangle
    # dist_mat is symmetric, sep_mat is symmetric.
    # overlap[i,j] = max(0, sep_mat[i,j] - dist_mat[i,j])
    # Sum of squares of overlaps for i < j
    
    # Mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    overlaps = np.maximum(0, sep_mat - dist_mat)
    pairwise_violation = np.sum(overlaps[mask]**2)
    
    obj -= beta * (boundary_violation + pairwise_violation)
    return obj

def _random_init():
    """Random initial configuration."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.01, 0.08, N)
    return x, y, r

def _grid_init():
    """Grid-based initial configuration, feasible."""
    side = 6 # 6x6 = 36 > 26
    # Place circles in a 6x6 grid, take first 26
    # Cell size 1/6
    # Radius approx 1/12 = 0.0833
    # To ensure feasibility, use slightly smaller radius
    r_init = 0.08
    x = []
    y = []
    r = []
    for i in range(side):
        for j in range(side):
            if len(r) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                x.append(cx)
                y.append(cy)
                r.append(r_init)
    return np.array(x), np.array(y), np.array(r)

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
    
    # Clamp positions to stay roughly in bounds
    x_new[idx] = np.clip(x_new[idx], r_new[idx], 1 - r_new[idx])
    y_new[idx] = np.clip(y_new[idx], r_new[idx], 1 - r_new[idx])
    
    return x_new, y_new, r_new

def _parallel_tempering(num_replicas=10, iterations_per_replica=2000, 
                         temp_base=0.02, temp_factor=1.5,
                         scale_pos=0.02, scale_r=0.005,
                         beta=500.0, seed=42):
    """Parallel tempering Monte Carlo for circle packing."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize replicas
    # One replica starts from grid init, others random
    states = []
    objectives = []
    
    # Replica 0: Grid init
    x, y, r = _grid_init()
    states.append((x, y, r))
    objectives.append(_penalized_objective(x, y, r, beta))
    
    # Other replicas: Random init
    for _ in range(num_replicas - 1):
        x, y, r = _random_init()
        states.append((x, y, r))
        objectives.append(_penalized_objective(x, y, r, beta))
    
    # Temperatures (geometric progression)
    # Lower temperature -> more exploitation (colder)
    # Higher temperature -> more exploration (hotter)
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

def _local_optimize(x, y, r, max_iters=50):
    """Local optimization to improve a feasible configuration by expanding radii."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    current_obj = _objective(x, y, r)
    
    for it in range(max_iters):
        improved = False
        # Try to increase each radius slightly
        for i in range(N):
            # Try a few small increments
            for dr in [0.001, 0.0005, 0.0002]:
                r_test = r.copy()
                r_test[i] += dr
                if _feasible(x, y, r_test):
                    r = r_test
                    current_obj = _objective(x, y, r)
                    improved = True
                    break
        if not improved:
            break
    
    return x, y, r, current_obj

def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    # Run parallel tempering
    # Increased iterations and replicas for better convergence
    best_state, best_obj = _parallel_tempering(
        num_replicas=10,
        iterations_per_replica=3000,
        temp_base=0.02,
        temp_factor=1.5,
        scale_pos=0.02,
        scale_r=0.005,
        beta=500.0,
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
    x, y, r, obj = _local_optimize(x, y, r, max_iters=50)
    
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
