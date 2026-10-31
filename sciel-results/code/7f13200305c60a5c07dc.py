"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _pairwise_distances(centers):
    """Compute pairwise distance matrix for centers (n, 2)."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

def _objective(params):
    """Objective: maximize sum of radii (minimize negative sum).
    params: first 2N are centers (x,y), last N are radii.
    Constraints handled via penalty for overlaps and boundary violations.
    """
    n = len(params) // 3
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Boundary penalty: circles must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    boundary_violations = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Lower bound violations
        if x - r < 0:
            boundary_violations += (x - r) ** 2
        if y - r < 0:
            boundary_violations += (y - r) ** 2
        # Upper bound violations
        if x + r > 1:
            boundary_violations += (x + r - 1) ** 2
        if y + r > 1:
            boundary_violations += (y + r - 1) ** 2
    
    # Overlap penalty: for each pair, distance >= r_i + r_j
    overlap_penalty = 0.0
    dist_mat = _pairwise_distances(centers)
    for i in range(n):
        for j in range(i+1, n):
            min_dist = radii[i] + radii[j]
            if dist_mat[i, j] < min_dist:
                overlap_penalty += (min_dist - dist_mat[i, j]) ** 2
    
    return -(np.sum(radii)) + 100.0 * (boundary_violations + overlap_penalty)

def _objective_and_grad(params):
    """Objective with gradient approximation for L-BFGS-B."""
    n = len(params) // 3
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    obj_val = _objective(params)
    
    # Numerical gradient
    grad = np.zeros_like(params)
    eps = 1e-6
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        grad[i] = (_objective(params_plus) - obj_val) / eps
        params[i] = params_plus[i]  # restore
    
    return obj_val, grad

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    init_params = []
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = 0.5 / side - 0.01
        init_params.extend([x, y, r])
    init_params = np.array(init_params)
    
    best_params = init_params.copy()
    best_value = np.sum(init_params[2*N:2*N+N])  # sum of radii
    
    # Update best if initial is valid
    def _check_validity(params):
        n = len(params) // 3
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-8 or x + r > 1 + 1e-8 or y - r < -1e-8 or y + r > 1 + 1e-8:
                return False
        # Check overlaps
        dist_mat = _pairwise_distances(centers)
        for i in range(n):
            for j in range(i+1, n):
                if dist_mat[i, j] < radii[i] + radii[j] - 1e-8:
                    return False
        return True
    
    if _check_validity(best_params):
        best_value = np.sum(best_params[2*N:2*N+N])
    else:
        best_value = 0.0
    
    # Try multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        if restart == 0:
            current_params = init_params.copy()
        else:
            # Random perturbation of the initial grid
            current_params = init_params.copy()
            noise = np.random.normal(0, 0.05, size=current_params.shape)
            current_params += noise
            # Clamp to reasonable bounds
            for i in range(N):
                current_params[2*i] = np.clip(current_params[2*i], 0.01, 0.99)
                current_params[2*i+1] = np.clip(current_params[2*i+1], 0.01, 0.99)
                current_params[2*i+2] = np.clip(current_params[2*i+2], 0.001, 0.2)
        
        # Optimization
        result = minimize(_objective, current_params, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() > deadline - safety_margin:
            break
        
        if result.fun < _objective(best_params):
            # Check validity
            if _check_validity(result.x):
                best_params = result.x.copy()
                best_value = np.sum(result.x[2*N:2*N+N])
        
        # Also try to polish the best solution found so far
        if time.time() > deadline - safety_margin:
            break
        
        # Polish pass with tighter tolerance
        result2 = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if _check_validity(result2.x):
            val = np.sum(result2.x[2*N:2*N+N])
            if val > best_value:
                best_params = result2.x.copy()
                best_value = val
    
    # Final check and return
    if not _check_validity(best_params):
        # Fallback to a known valid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
        return np.array(cells)
    
    # Format output: (x, y, radius)
    output = np.zeros((N, 3))
    for i in range(N):
        output[i, 0] = best_params[2*i]
        output[i, 1] = best_params[2*i+1]
        output[i, 2] = best_params[2*i+2]
    
    return output
# EVOLVE-BLOCK-END
