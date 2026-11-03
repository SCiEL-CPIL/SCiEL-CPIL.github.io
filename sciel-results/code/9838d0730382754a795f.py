"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Safety margin: 150s limit, leave 5s for overhead
_DEADLINE = 145.0

def _log_energy_precise(points):
    """Compute log(1/d) energy using math.fsum for high precision."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    # Use fsum for accuracy
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Energy and gradient for L-BFGS-B.
    x is flat array of shape (N*3,).
    Normalizes x to sphere internally.
    Returns (energy, gradient_flat).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    # Pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    
    # Mask diagonal to avoid division by zero
    mask = ~np.eye(n, dtype=bool)
    dists_sq[mask] = dists_sq[mask]
    dists_sq[~mask] = 1.0 # Placeholder, will be overwritten or ignored
    
    # Calculate weights 1/d^2
    # Avoid division by zero
    inv_dists_sq = np.zeros_like(dists_sq)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    # Energy: -sum_{i<j} log(d_ij)
    # d_ij = sqrt(dists_sq_ij)
    # log(d_ij) = 0.5 * log(dists_sq_ij)
    iu = np.triu_indices(n, k=1)
    # Use np.sum for speed in optimization loop
    energy = -0.5 * np.sum(np.log(dists_sq[iu]))
    
    # Gradient calculation
    # grad_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # grad_i = x_i * sum_{j!=i} 1/d_ij^2 - sum_{j!=i} x_j / d_ij^2
    # Note: The term (x_i - x_j) / d^2 can be seen as:
    # sum_j W_ij x_j - x_i sum_j W_ij
    # where W_ij = 1/d^2 for i!=j, 0 for i=j.
    # Wait, the gradient of -log(d) is (x_i - x_j)/d^2.
    # Let's verify signs.
    # E = - sum log(d). dE/dd = -1/d. dd/dx_i = (x_i-x_j)/d.
    # dE/dx_i = -1/d * (x_i-x_j)/d = -(x_i-x_j)/d^2 = (x_j-x_i)/d^2.
    # So grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    #            = sum_{j!=i} x_j / d_ij^2 - x_i sum_{j!=i} 1/d_ij^2
    
    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project to tangent space to ensure we stay on manifold logic
    # Although L-BFGS handles the projection via the objective, 
    # providing the tangential gradient is more stable.
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    return energy, grads_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    
    if time.time() > deadline - 1.0:
        return points, _log_energy_precise(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        
        if time.time() > deadline - 1.0:
            return points, _log_energy_precise(points)
            
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        # Use precise energy for comparison
        e_check = _log_energy_precise(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy_precise(points)

def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm

def _basin_hopping(init_points, deadline, rng, max_iter=100, T_init=0.1):
    """
    Basin hopping loop.
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.95

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        # Larger perturbations help escape deep local minima
        n_perturb = rng.randint(3, 10)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=100)

        # Metropolis criterion (accept if better or with prob exp(-dE/T))
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy

def _polish_high_precision(pts, max_time):
    """High-precision final polish with L-BFGS at tight tolerances."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy_precise(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy_precise(pts)

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci
    x_fib = _fibonacci_sphere(N)
    initializations.append(x_fib)
    
    # 2. Random seeds
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        x_rand = _random_uniform_sphere(N, rng)
        initializations.append(x_rand)

    best_x = None
    best_e = float('inf')
    
    # Phase 1: Local minimize each initialization
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
            
        try:
            x_opt, e_opt = _local_minimize(x_init, deadline, maxiter=300)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
            
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        best_e = _log_energy_precise(best_x)
        
    # Phase 2: Basin Hopping from the best local minimum
    if time.time() < deadline - 10:
        try:
            x_bh, e_bh = _basin_hopping(best_x, deadline, rng, max_iter=50, T_init=0.15)
            if e_bh < best_e:
                best_x = x_bh
                best_e = e_bh
        except Exception:
            pass
            
    # Phase 3: Final High-Precision Polish
    if time.time() < deadline - 2:
        try:
            x_final, e_final = _polish_high_precision(best_x, max_time=deadline - time.time() - 1)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
        except Exception:
            pass
            
    # Final validation and projection
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close (check for min distance > 1e-9)
    # If points are too close, nudge them
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                rng_fix = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng_fix.randn(3)
                best_x = _project_to_sphere(best_x)
                
    # Check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END
