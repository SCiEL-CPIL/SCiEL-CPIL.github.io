"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(C):
    """Compute the actual packing score (radius)."""
    N = C.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    pair_dists = dist[i_idx, j_idx]
    radii = np.linalg.norm(C, axis=1)
    bound_gaps = 1.0 - radii
    return np.min(np.concatenate([pair_dists / 2.0, bound_gaps]))

def _violation_potential(C, r, power=4):
    """
    Compute a smooth violation potential for radius r.
    Returns 0 if all constraints are satisfied with margin.
    Positive if any constraint is violated.
    """
    N = C.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Pairwise constraints: |C_i - C_j| >= 2r
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    pair_dists = dist[i_idx, j_idx]
    pair_violations = np.maximum(0, 2*r - pair_dists)
    
    # Boundary constraints: |C_i| <= 1 - r
    radii = np.linalg.norm(C, axis=1)
    bound_violations = np.maximum(0, radii - (1 - r))
    
    # Sum of powered violations (smooth approximation)
    return np.sum(pair_violations**power) + np.sum(bound_violations**power)

def _violation_grad(C, r, power=4):
    """Compute gradient of violation potential."""
    N = C.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    grad = np.zeros_like(C)
    
    # Pairwise
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    pair_dists = dist[i_idx, j_idx]
    
    mask = pair_dists < 2*r
    if np.any(mask):
        idx_m = np.where(mask)[0]
        i_m = i_idx[idx_m]
        j_m = j_idx[idx_m]
        d_m = pair_dists[idx_m]
        diff_m = C[i_m] - C[j_m]
        # d/dC_i (2r - |diff|)^power = -power*(2r-|diff|)^(power-1) * diff/|diff|
        viol = 2*r - d_m
        coeff = power * (viol**(power-1)) / d_m
        contrib = coeff[:, np.newaxis] * diff_m
        np.add.at(grad, i_m, contrib)
        np.add.at(grad, j_m, -contrib)
    
    # Boundary
    radii = np.linalg.norm(C, axis=1)
    mask_b = radii > (1 - r)
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        r_b = radii[idx_b]
        viol_b = r_b - (1 - r)
        coeff_b = power * viol_b**(power-1) / r_b
        grad[idx_b] += coeff_b[:, np.newaxis] * C[idx_b]
    
    return grad

def _hex_init(n_points, r_target=0.12):
    """Generate initial points on a hexagonal lattice."""
    spacing = 2 * r_target * 1.05
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y < (1 - r_target)**2:
                points.append([x, y])
    if len(points) >= n_points:
        return np.array(points[:n_points])
    # Fallback: random
    angles = np.random.uniform(0, 2*np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points)) * (1 - r_target)
    return np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

def _random_init(n_points, r_target=0.12):
    """Random initialization."""
    angles = np.random.uniform(0, 2*np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points)) * (1 - r_target) * 0.9
    return np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

def _optimize_for_radius(C_init, r, deadline, power=4):
    """
    Try to find a configuration with all circles of radius r fitting.
    Uses L-BFGS-B on the violation potential.
    Returns (success, C_optimized, final_violation)
    """
    def obj(x):
        return _violation_potential(x.reshape(N, 2), r, power)
    
    def grad(x):
        return _violation_grad(x.reshape(N, 2), r, power).flatten()
    
    x0 = C_init.flatten()
    
    result = minimize(
        obj, x0, jac=grad,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
    )
    
    C_opt = result.x.reshape(N, 2)
    final_viol = _violation_potential(C_opt, r, power)
    
    # Check if feasible
    score = _compute_score(C_opt)
    success = score >= r - 1e-9
    
    return success, C_opt, final_viol

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_C = None
    
    # Binary search parameters
    r_low = 0.0
    r_high = 0.15  # Upper bound estimate
    
    # First, get a decent starting configuration
    C_start = _hex_init(N, r_target=0.12)
    best_score = _compute_score(C_start)
    best_C = C_start.copy()
    
    # Do a few random restarts to find a good initial solution
    for _ in range(3):
        C_rand = _random_init(N, r_target=0.12)
        # Quick local optimization
        def obj(x):
            return _violation_potential(x.reshape(N, 2), 0.12, power=4)
        def grad(x):
            return _violation_grad(x.reshape(N, 2), 0.12, power=4).flatten()
        result = minimize(obj, C_rand.flatten(), jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10})
        C_opt = result.x.reshape(N, 2)
        s = _compute_score(C_opt)
        if s > best_score:
            best_score = s
            best_C = C_opt.copy()
    
    # Binary search on radius
    num_iterations = 20  # About 20 bisection steps
    for it in range(num_iterations):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to find a feasible configuration at radius r_mid
        # Use best_C as initial guess
        success, C_opt, viol = _optimize_for_radius(best_C, r_mid, deadline)
        
        if success:
            r_low = r_mid
            best_C = C_opt.copy()
            best_score = _compute_score(C_opt)
        else:
            r_high = r_mid
    
    # Final refinement: try to squeeze out more radius from best_C
    # Use gradient ascent on the actual score
    C = best_C.copy()
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Compute current score and identify binding constraints
        i_idx, j_idx = np.triu_indices(N, k=1)
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        pair_dists = dist[i_idx, j_idx]
        radii = np.linalg.norm(C, axis=1)
        bound_gaps = 1.0 - radii
        
        all_gaps = np.concatenate([pair_dists / 2.0, bound_gaps])
        min_gap = np.min(all_gaps)
        
        # Move points to increase the minimum gap
        # Simple approach: move each point away from its closest neighbor/boundary
        grad = np.zeros_like(C)
        
        # For pairs
        min_pair_dists = pair_dists.copy()
        for k in range(len(i_idx)):
            if min_pair_dists[k] < min_gap * 1.5:  # Only consider near-binding
                i, j = i_idx[k], j_idx[k]
                d = dist[i, j]
                if d > 1e-10:
                    unit = (C[i] - C[j]) / d
                    grad[i] += unit * 0.5
                    grad[j] -= unit * 0.5
        
        # For boundaries
        for i in range(N):
            if bound_gaps[i] < min_gap * 1.5:
                r_i = radii[i]
                if r_i > 1e-10:
                    grad[i] -= (C[i] / r_i)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        step_size = 0.01
        C_new = C + step_size * grad
        
        # Check if improvement
        new_score = _compute_score(C_new)
        if new_score > min_gap + 1e-10:
            C = C_new
            if new_score > best_score:
                best_score = new_score
                best_C = C.copy()
        else:
            step_size *= 0.5
    
    return best_C
# EVOLVE-BLOCK-END
