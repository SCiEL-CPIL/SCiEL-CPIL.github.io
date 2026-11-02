"""Circle packing in unit square via parallel tempering Monte Carlo."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np

def _penalty_value(x, y, r, beta=1000.0):
    """Calculate penalty for constraint violations. Lower is better (0 is feasible)."""
    # Boundary violations
    bx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
    by = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    pen = np.sum(bx**2) + np.sum(by**2)
    
    # Pairwise overlaps
    # Compute distance matrix efficiently
    # Using broadcasting for N=26 is fine (26x26 matrix)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    req_dist = r[:, np.newaxis] + r[np.newaxis, :]
    overlap = np.maximum(0, req_dist - dist)
    # Exclude diagonal
    np.fill_diagonal(overlap, 0)
    
    pen += np.sum(overlap**2)
    
    return beta * pen

def _is_feasible(x, y, r, tol=1e-7):
    """Check feasibility."""
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    req_dist = r[:, np.newaxis] + r[np.newaxis, :]
    # Check if any distance is less than sum of radii
    if np.any(dist < req_dist - tol):
        return False
    return True

def _random_init():
    """Random initial configuration."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    # Start with small radii to ensure feasibility initially
    r = np.random.uniform(0.01, 0.05, N)
    return x, y, r

def _make_move(x, y, r, scale_pos=0.02, scale_r=0.005):
    """Make a random move on a random circle."""
    x_new = x.copy()
    y_new = y.copy()
    r_new = r.copy()
    
    idx = np.random.randint(0, N)
    
    # Move position
    x_new[idx] += np.random.normal(0, scale_pos)
    y_new[idx] += np.random.normal(0, scale_pos)
    
    # Move radius
    r_new[idx] += np.random.normal(0, scale_r)
    r_new[idx] = np.clip(r_new[idx], 0.005, 0.45)
    
    # Clamp positions to stay within [0,1] roughly, respecting radius
    # This helps keep the chain in a sane region
    x_new[idx] = np.clip(x_new[idx], 0, 1)
    y_new[idx] = np.clip(y_new[idx], 0, 1)
    
    return x_new, y_new, r_new

def _parallel_tempering(num_replicas=10, iterations_per_replica=2000, 
                         temp_base=0.01, temp_factor=2.0,
                         scale_pos=0.02, scale_r=0.005,
                         beta=1000.0, seed=42):
    """Parallel tempering Monte Carlo for circle packing."""
    np.random.seed(seed)
    
    # Initialize replicas
    states = []
    penalties = []
    for _ in range(num_replicas):
        x, y, r = _random_init()
        states.append((x, y, r))
        penalties.append(_penalty_value(x, y, r, beta))
    
    # Temperatures
    temps = np.array([temp_base * (temp_factor ** i) for i in range(num_replicas)])
    
    best_state = None
    best_obj = -np.inf
    
    # Check initial states
    for rep in range(num_replicas):
        x, y, r = states[rep]
        if _is_feasible(x, y, r):
            obj = np.sum(r)
            if obj > best_obj:
                best_obj = obj
                best_state = (x.copy(), y.copy(), r.copy())
    
    # Main loop
    for it in range(iterations_per_replica):
        # For each replica, make a move
        for rep in range(num_replicas):
            x, y, r = states[rep]
            x_new, y_new, r_new = _make_move(x, y, r, scale_pos, scale_r)
            
            pen_old = penalties[rep]
            pen_new = _penalty_value(x_new, y_new, r_new, beta)
            
            # Metropolis acceptance on penalty (minimization)
            # We want to accept if penalty decreases.
            # delta = pen_new - pen_old. If delta < 0, accept.
            # Accept with prob exp(-delta / T)
            delta = pen_new - pen_old
            if delta <= 0 or np.random.random() < np.exp(-delta / temps[rep]):
                states[rep] = (x_new, y_new, r_new)
                penalties[rep] = pen_new
                
                # Check if feasible and update best
                if _is_feasible(x_new, y_new, r_new):
                    obj_feas = np.sum(r_new)
                    if obj_feas > best_obj:
                        best_obj = obj_feas
                        best_state = (x_new.copy(), y_new.copy(), r_new.copy())
        
        # Swap adjacent replicas
        for rep in range(num_replicas - 1):
            rep_a = rep
            rep_b = rep + 1
            
            x_a, y_a, r_a = states[rep_a]
            x_b, y_b, r_b = states[rep_b]
            pen_a = penalties[rep_a]
            pen_b = penalties[rep_b]
            T_a = temps[rep_a]
            T_b = temps[rep_b]
            
            # Swap probability based on energy difference
            # E = penalty. 
            # Accept swap with prob exp(-(E_a/T_a + E_b/T_b) - (E_b/T_a + E_a/T_b))
            # = exp(-(E_a - E_b)(1/T_a - 1/T_b))
            delta = (pen_a - pen_b) * (1.0 / T_a - 1.0 / T_b)
            if delta <= 0 or np.random.random() < np.exp(-delta):
                states[rep_a] = (x_b, y_b, r_b)
                states[rep_b] = (x_a, y_a, r_a)
                penalties[rep_a] = pen_b
                penalties[rep_b] = pen_a
    
    return best_state, best_obj

def _local_optimize_scipy(x, y, r):
    """Use scipy minimize to refine positions and radii."""
    # Flatten variables: x (N), y (N), r (N)
    def objective(v):
        x_v = v[:N]
        y_v = v[N:2*N]
        r_v = v[2*N:]
        
        # Penalty for infeasibility
        pen = _penalty_value(x_v, y_v, r_v, beta=10000.0)
        
        # Maximize sum of radii -> Minimize -sum(r)
        return pen - np.sum(r_v)
    
    v0 = np.concatenate([x, y, r])
    
    # Bounds
    # x, y in [0, 1]
    # r in [0.001, 0.45]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0.001, 0.45)] * N
    
    # Use SLSQP or COBYLA. SLSQP is generally good for this.
    res = minimize(objective, v0, method='SLSQP', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
    
    if res.success:
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        r_opt = res.x[2*N:]
        
        # Check feasibility
        if _is_feasible(x_opt, y_opt, r_opt):
            return x_opt, y_opt, r_opt
    
    return x, y, r

def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    # Run parallel tempering
    best_state, best_obj = _parallel_tempering(
        num_replicas=10,
        iterations_per_replica=2000,
        temp_base=0.01,
        temp_factor=2.0,
        scale_pos=0.02,
        scale_r=0.005,
        beta=1000.0,
        seed=42
    )
    
    if best_state is None:
        # Fallback: try random feasible configs
        for _ in range(100):
            x, y, r = _random_init()
            if _is_feasible(x, y, r):
                best_state = (x, y, r)
                break
    
    if best_state is None:
        # Last resort: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r = best_state
    
    # Local optimization using Scipy
    x, y, r = _local_optimize_scipy(x, y, r)
    
    # Final feasibility check and repair
    if not _is_feasible(x, y, r):
        # Try to fix by shrinking radii slightly
        for _ in range(100):
            r = r * 0.99
            if _is_feasible(x, y, r):
                break
        # If still not feasible, try moving positions slightly? 
        # For now, just return if feasible, else try a few more local steps
        if not _is_feasible(x, y, r):
             # Shrink more aggressively
             r = r * 0.9
             if not _is_feasible(x, y, r):
                 r = r * 0.8
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
