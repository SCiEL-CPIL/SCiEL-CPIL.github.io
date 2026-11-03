import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def _max_radius_for_positions(x, y):
    """Compute the maximum radius for each circle given positions."""
    n = len(x)
    r = np.empty(n)
    for i in range(n):
        # Distance to boundaries
        max_r_i = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                # r_i + r_j <= d, so r_i <= d - r_j
                # We iterate: start with r_j = 0, then update
                max_r_i = min(max_r_i, d - 0.0)  # placeholder
        # Better: solve iteratively
        r[i] = max_r_i
    # Iterative refinement
    for _ in range(10):
        for i in range(n):
            max_r_i = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            for j in range(n):
                if i != j:
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    max_r_i = min(max_r_i, d - r[j])
            r[i] = max(0.0, max_r_i)
    return r

def _objective(z):
    """Objective: negative sum of radii + penalties for violations."""
    z = z.reshape(N, 3)
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    
    sum_r = np.sum(r)
    
    # Overlap penalty
    dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    r_i = r[iu[0]]
    r_j = r[iu[1]]
    overlaps = np.maximum(0, r_i + r_j - d_ij)
    penalty_overlap = np.sum(overlaps**2)
    
    # Boundary penalty
    pen_b = 0.0
    pen_b += np.sum(np.maximum(0, r - x)**2)
    pen_b += np.sum(np.maximum(0, r - (1 - x))**2)
    pen_b += np.sum(np.maximum(0, r - y)**2)
    pen_b += np.sum(np.maximum(0, r - (1 - y))**2)
    
    penalty = penalty_overlap + pen_b
    return -sum_r + 100.0 * penalty

def _is_valid(z, tol=1e-6):
    """Check if solution is valid (no overlaps, within bounds)."""
    z = z.reshape(N, 3)
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    
    # Boundary check
    if np.any(r < -tol) or np.any(r > 0.5 + tol):
        return False
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol):
        return False
    
    # Overlap check
    dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    r_i = r[iu[0]]
    r_j = r[iu[1]]
    if np.any(r_i + r_j - d_ij > tol):
        return False
    
    return True

def _initial_grid():
    """5x5 grid + 1 extra in corner."""
    inits = []
    for i in range(5):
        for j in range(5):
            inits.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0, 0.1])
    inits.append([0.1, 0.1, 0.1])
    return np.array(inits).flatten()

def _initial_random(seed=42):
    """Random initialization."""
    np.random.seed(seed)
    x = np.random.rand(N)
    y = np.random.rand(N)
    r = np.full(N, 0.05)
    return np.column_stack((x, y, r)).flatten()

def _simulated_annealing(z0, start_temp=0.05, end_temp=0.001, n_iter=5000, seed=42, time_limit=30.0):
    """Simulated annealing to optimize circle packing."""
    np.random.seed(seed)
    z = z0.copy()
    current_obj = _objective(z)
    best_obj = current_obj
    best_z = z.copy()
    
    start_time = time.time()
    n_accepted = 0
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Cooling schedule (exponential)
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Generate perturbation
        z_new = z.copy()
        # Move a random circle
        idx = np.random.randint(N)
        z_new[idx * 3] += np.random.normal(0, 0.02)
        z_new[idx * 3 + 1] += np.random.normal(0, 0.02)
        # Optionally adjust radius
        if np.random.random() < 0.3:
            z_new[idx * 3 + 2] += np.random.normal(0, 0.01)
        
        # Clamp to bounds
        z_new[0::3] = np.clip(z_new[0::3], 0, 1)
        z_new[1::3] = np.clip(z_new[1::3], 0, 1)
        z_new[2::3] = np.clip(z_new[2::3], 1e-6, 0.5)
        
        new_obj = _objective(z_new)
        delta = new_obj - current_obj
        
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            z = z_new
            current_obj = new_obj
            n_accepted += 1
            if current_obj < best_obj:
                best_obj = current_obj
                best_z = z.copy()
    
    return best_z, best_obj

def _local_search(z, max_iter=100, time_limit=10.0):
    """Local refinement using gradient-based optimization."""
    start_time = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, 0.5))
    
    try:
        res = minimize(_objective, z, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            return res.x, res.fun
    except Exception:
        pass
    
    return z, _objective(z)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_solution = None
    
    # Generate diverse initializations
    inits = [
        _initial_grid(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(456),
    ]
    
    # Run SA + local search from each init
    for init_idx, z0 in enumerate(inits):
        if time.time() - start_time > timeout:
            break
        
        # Simulated annealing
        sa_time = min(25.0, max(5.0, (timeout - (time.time() - start_time)) / 2))
        z_sa, obj_sa = _simulated_annealing(z0, start_temp=0.05, end_temp=0.001, 
                                            n_iter=8000, seed=42 + init_idx, time_limit=sa_time)
        
        # Local refinement
        ls_time = min(10.0, max(2.0, (timeout - (time.time() - start_time)) / 2))
        z_ls, obj_ls = _local_search(z_sa, max_iter=50, time_limit=ls_time)
        
        # Validate and update best
        z_check = z_ls.reshape(N, 3)
        x, y, r = z_check[:, 0], z_check[:, 1], z_check[:, 2]
        
        # Check validity
        valid = True
        if np.any(r < -1e-6) or np.any(x - r < -1e-6) or np.any(1 - x - r < -1e-6) or \
           np.any(y - r < -1e-6) or np.any(1 - y - r < -1e-6):
            valid = False
        
        if valid:
            dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
            iu = np.triu_indices(N, k=1)
            d_ij = dist[iu]
            r_i, r_j = r[iu[0]], r[iu[1]]
            if np.any(r_i + r_j - d_ij > 1e-5):
                valid = False
        
        if valid:
            sum_r = np.sum(r)
            if sum_r > best_score:
                best_score = sum_r
                best_solution = z_ls.copy()
    
    # Fallback if no valid solution found
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
