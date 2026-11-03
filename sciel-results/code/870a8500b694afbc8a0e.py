"""
Optimized Logarithmic Energy Minimization for 150 Points on Sphere
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

def _log_energy(points):
    """
    Compute log(1/d) energy for points on unit sphere.
    Returns a scalar float.
    """
    n = len(points)
    # Compute pairwise squared distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    
    # We need distances for i < j
    # Using triu indices to avoid double counting and self-distances
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    # Energy is sum of log(1/d) = -sum(log(d))
    # Use math.fsum for high precision summation as required by the task
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Calculate energy and gradient for L-BFGS-B.
    x is a flat array of shape (N*3,).
    """
    points = x.reshape(N, 3)
    
    # Normalize points to sphere (L-BFGS-B might drift)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero for self
    
    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Note: d_ij^2 = |x_i - x_j|^2
    
    # Weights matrix W where W[i,j] = 1 / d_ij^2 for i != j, else 0
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    # grad_i = x_i * sum_j W[i,j] - sum_j W[i,j] * x_j
    weighted_sum = weights @ points
    sum_weights = np.sum(weights, axis=1)
    
    grads = points * sum_weights[:, np.newaxis] - weighted_sum
    
    # Project gradient to tangent space: grad_tan = grad - (grad . p) * p
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    energy = _log_energy(points)
    
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
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _local_minimize(points, deadline, maxiter=500, tol=1e-12):
    """
    Run L-BFGS-B to find local minimum.
    Returns (points, energy).
    """
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': tol, 'gtol': tol, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

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

def _simulated_annealing_with_polish(points, deadline, rng, max_iter=2000, T_start=0.1, T_end=1e-6, polish_freq=500):
    """
    Simulated Annealing with periodic local minimization (polish).
    """
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    n_iter = max_iter
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Linear cooling
        T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        
        # Move one random point
        idx = rng.randint(N)
        tangent = _tangent_vector(current[idx], rng)
        # Step size scales with T
        step_size = T * rng.uniform(0.5, 1.5)
        new_point = current[idx] + step_size * tangent
        new_point = new_point / np.linalg.norm(new_point)
        
        new_points = current.copy()
        new_points[idx] = new_point
        
        # Calculate new energy
        # Optimization: We can update energy incrementally, but for N=150, full recalc is fast enough
        # Full recalc is O(N^2) = 22500 ops, which is trivial for NumPy
        new_energy = _log_energy(new_points)
        
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
        
        # Periodic Polish: Run local minimization to settle into basin
        if (it + 1) % polish_freq == 0:
            if time.time() > deadline - 2.0:
                break
            polished_points, polished_energy = _local_minimize(current, deadline, maxiter=100, tol=1e-10)
            if polished_energy < current_energy:
                current = polished_points
                current_energy = polished_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best = current.copy()
    
    return best, best_energy

def _basin_hopping(points, deadline, rng, max_iter=50):
    """
    Basin Hopping with large perturbations and local minimization.
    """
    current_points, current_energy = _local_minimize(points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.5
    T_min = 1e-5
    cooling = 0.95
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break
            
        # Perturb: Move a random subset of points significantly
        n_perturb = rng.randint(3, 10)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * rng.uniform(0.5, 2.0)
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point
        
        # Local minimize
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=200)
        
        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= cooling

    return best_points, best_energy

def _final_polish(points, deadline):
    """
    Final high-precision polishing.
    """
    if time.time() > deadline - 2.0:
        return points, _log_energy(points)
    
    # Tighter tolerance
    p1, e1 = _local_minimize(points, deadline, maxiter=2000, tol=1e-14)
    
    # If we have time, do another pass
    if time.time() < deadline - 1.0:
         p2, e2 = _local_minimize(p1, deadline, maxiter=1000, tol=1e-15)
         if e2 < e1:
             return p2, e2
             
    return p1, e1

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initializations
    inits = []
    
    # Fibonacci sphere (standard)
    inits.append(_fibonacci_sphere(N))
    
    # Perturbed Fibonacci
    for k in range(5):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.1
        inits.append(_project_to_sphere(perturbed))
        
    # Random points
    for seed in range(10):
        r = np.random.RandomState(seed * 13 + 7)
        inits.append(_random_uniform_sphere(N, r))

    # 2. Local minimize all initializations to find good basins
    candidates = []
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=500, tol=1e-12)
        candidates.append((e_res, p_res))
        
    candidates.sort(key=lambda x: x[0])
    
    best_energy = candidates[0][0]
    best_points = candidates[0][1].copy()
    
    # 3. Apply Basin Hopping to the best few candidates
    n_bh = min(8, len(candidates))
    for i in range(n_bh):
        if time.time() > deadline - 25.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 123)
        bh_points, bh_energy = _basin_hopping(candidates[i][1], deadline, bh_rng, max_iter=40)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()
            
    # 4. Apply Simulated Annealing with Polish to the best configuration
    if time.time() < deadline - 15.0:
        sa_rng = np.random.RandomState(999)
        # Use a long SA run
        sa_points, sa_energy = _simulated_annealing_with_polish(
            best_points, deadline, sa_rng, 
            max_iter=5000, T_start=0.08, T_end=1e-8, polish_freq=400
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()
            
    # 5. Final Polish with tighter tolerance
    if time.time() < deadline - 3.0:
        polished, polished_energy = _final_polish(best_points, deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_points = polished
    
    best_points = _project_to_sphere(best_points)
    
    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
