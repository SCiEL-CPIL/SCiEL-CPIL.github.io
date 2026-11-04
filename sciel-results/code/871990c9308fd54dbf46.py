"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the max common radius for centres c (N,3) in [-0.5,0.5]^3."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # Boundary distance
    d_bound = 0.5 - np.abs(c).max(axis=1)
    r_bound = d_bound.min()
    if r_bound <= 0:
        return -1.0
    # Pairwise distances
    # Use scipy cdist if available, else manual
    try:
        from scipy.spatial.distance import cdist
        D = cdist(c, c)
    except ImportError:
        D = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    # Set diagonal to inf
    np.fill_diagonal(D, np.inf)
    d_min = D.min()
    r_pair = d_min / 2.0
    return min(r_pair, r_bound)


def _soft_min(x, beta=1000.0):
    """Soft minimum using log-sum-exp trick."""
    m = x.min()
    return m + np.log(np.exp(x - m).sum()) / len(x)


def _energy_obj(c, r_target, beta=500.0):
    """
    Objective: minimize violations of the constraint that all pairwise distances
    and boundary distances are >= 2*r_target.
    Uses a soft penalty.
    """
    c = np.asarray(c, dtype=np.float64)
    # Boundary violations
    d_bound = 0.5 - np.abs(c)  # (N,3)
    r_bound = d_bound.min(axis=1)  # (N,)
    # We want r_bound >= r_target, i.e., r_bound - r_target >= 0
    # Penalty for boundary: soft-min of (r_bound - r_target)
    # Actually, let's use a different approach: minimize sum of squared violations
    # But soft-min is smoother. Let's use:
    # For each sphere i, the constraint is min_k (0.5 - |c_ik|) >= r_target
    # This is equivalent to min_k (0.5 - |c_ik| - r_target) >= 0
    # Let's define g_i = min_k (0.5 - |c_ik| - r_target)
    # We want to minimize the soft-min of all g_i and all pairwise (|c_i-c_j|/2 - r_target)
    
    # Pairwise distances
    D = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    # For each pair, constraint is D_ij/2 >= r_target, i.e., D_ij/2 - r_target >= 0
    # Let's collect all constraints:
    # Boundary: for each i, min_k (0.5 - |c_ik| - r_target)
    # Pairwise: for each i<j, D_ij/2 - r_target
    
    # To keep it simple, let's just compute the actual radius and use a penalty
    # based on how much it falls short of r_target.
    # Actually, let's use the approach from the knowledge: target radius violation minimization.
    # Objective: max(0, r_target - r_actual) but smoothed.
    # r_actual = min(r_bound_min, r_pair_min)
    # Let's compute r_actual and return a smooth penalty.
    
    r_bound_min = (0.5 - np.abs(c).max(axis=1)).min()
    r_pair_min = D.min() / 2.0
    r_actual = min(r_bound_min, r_pair_min)
    
    # Smooth penalty: if r_actual < r_target, penalize
    # Use a quadratic penalty: 0.5 * max(0, r_target - r_actual)^2
    # But this is not differentiable at r_actual = r_target.
    # Use soft version: 0.5 * (r_target - r_actual + softplus(r_actual - r_target))^2 ... no.
    # Let's use: 0.5 * (r_target - r_actual)^2 when r_actual < r_target, else 0
    # Smooth: 0.5 * (r_target - r_actual + beta*softplus((r_target - r_actual)/beta))^2 ... 
    # Simpler: just use the raw violation squared, it's piecewise quadratic.
    # For BFGS, this should be okay with line search.
    
    if r_actual >= r_target:
        return 0.0
    else:
        return 0.5 * (r_target - r_actual) ** 2


def _energy_obj_grad(c, r_target):
    """Gradient of the energy objective."""
    c = np.asarray(c, dtype=np.float64)
    r_bound_min = (0.5 - np.abs(c).max(axis=1)).min()
    D = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    r_pair_min = D.min() / 2.0
    r_actual = min(r_bound_min, r_pair_min)
    
    if r_actual >= r_target:
        return np.zeros_like(c)
    
    violation = r_target - r_actual
    # Gradient of 0.5 * violation^2 w.r.t. c is violation * grad(violation)
    # violation = r_target - min(r_bound_min, r_pair_min)
    # grad(violation) = -grad(min(r_bound_min, r_pair_min))
    # We need to know which constraint is active.
    
    grad = np.zeros_like(c)
    
    if r_bound_min <= r_pair_min:
        # Active constraint is boundary
        # r_bound_min = min_i (0.5 - max_k |c_ik|)
        # Find the sphere i* that achieves the min
        i_star = np.argmin(0.5 - np.abs(c).max(axis=1))
        # For sphere i*, find the axis k* that achieves the max
        k_star = np.argmax(np.abs(c[i_star]))
        # grad of (0.5 - |c_{i*,k*}|) w.r.t. c_{i*,k*} is -sign(c_{i*,k*})
        # But we need grad of r_bound_min, which is min_i (0.5 - max_k |c_ik|)
        # d(r_bound_min)/dc_{i*,k*} = -sign(c_{i*,k*})
        # So d(violation)/dc_{i*,k*} = -(-sign(c_{i*,k*})) = sign(c_{i*,k*})
        # grad = violation * sign(c_{i*,k*}) e_{i*,k*}
        s = np.sign(c[i_star, k_star])
        if s == 0:
            s = 1.0
        grad[i_star, k_star] = violation * s
    else:
        # Active constraint is pairwise
        # Find the pair (i*, j*) that achieves the min distance
        i_star, j_star = np.unravel_index(D.argmin(), D.shape)
        # d(r_pair_min)/dc_{i*} = (c_{i*} - c_{j*}) / D_{i*j*}
        # d(r_pair_min)/dc_{j*} = (c_{j*} - c_{i*}) / D_{i*j*}
        d_ij = D[i_star, j_star]
        if d_ij < 1e-15:
            return np.zeros_like(c)
        vec = (c[i_star] - c[j_star]) / d_ij
        # grad(violation) = -grad(r_pair_min)
        # d(violation)/dc_{i*} = -vec
        # d(violation)/dc_{j*} = -(-vec) = vec
        grad[i_star] = -violation * vec
        grad[j_star] = violation * vec
    
    return grad


def construct():
    t_start = time.time()
    deadline = 130.0  # seconds, leaving 20s margin
    
    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    # Compute initial radius
    r_current = _compute_radius(c)
    r_best = r_current
    c_best = c.copy()
    
    # Target radius approach: start with a target slightly above current
    r_target = r_current + 0.001
    
    # Use BFGS to minimize the violation
    from scipy.optimize import minimize
    
    def obj(c_flat):
        c = c_flat.reshape(N, 3)
        return _energy_obj(c, r_target)
    
    def obj_grad(c_flat):
        c = c_flat.reshape(N, 3)
        return _energy_obj_grad(c, r_target).flatten()
    
    # Iteratively increase target radius
    for it in range(50):
        if time.time() > deadline:
            break
        
        res = minimize(obj, c.flatten(), jac=obj_grad, method='BFGS',
                       options={'maxiter': 100, 'gtol': 1e-8})
        
        c_new = res.x.reshape(N, 3)
        r_new = _compute_radius(c_new)
        
        if r_new > r_best:
            r_best = r_new
            c_best = c_new.copy()
        
        # Increase target
        r_target = max(r_new + 0.0005, r_target * 1.01)
        
        if not np.all(np.isfinite(c_new)):
            break
    
    # Final polish: try to squeeze out more precision
    # Use a tighter tolerance and more iterations
    r_target = r_best + 0.0001
    for it in range(20):
        if time.time() > deadline:
            break
        
        res = minimize(obj, c_best.flatten(), jac=obj_grad, method='BFGS',
                       options={'maxiter': 200, 'gtol': 1e-10})
        
        c_new = res.x.reshape(N, 3)
        r_new = _compute_radius(c_new)
        
        if r_new > r_best:
            r_best = r_new
            c_best = c_new.copy()
        
        r_target = r_new + 0.0001
    
    # Ensure validity
    if not np.all(np.isfinite(c_best)):
        return c.copy()
    
    # Check if centres are inside the cube
    if np.any(np.abs(c_best) > 0.5 + 1e-10):
        # Clip to boundary
        c_best = np.clip(c_best, -0.5, 0.5)
    
    return c_best

# EVOLVE-BLOCK-END
