"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    # Compute squared distances more efficiently
    # |xi - xj|^2 = |xi|^2 + |xj|^2 - 2*xi.xj = 2 - 2*xi.xj for unit vectors
    dots = points @ points.T
    dists_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dists_sq, 1.0)
    # Clip to avoid log(0) or negative due to floating point
    dists_sq = np.maximum(dists_sq, 1e-30)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    # Use math.fsum for accurate summation as specified
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    # Use dot product formula for unit vectors
    dots = points @ points.T
    dists_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dists_sq, 1.0)
    dists_sq = np.maximum(dists_sq, 1e-30)
    
    # Gradient of E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2  (note sign: E = -sum log(d), so grad points toward decreasing d)
    # Actually: E = -sum_{i<j} log(d_ij), dE/dx_i = -sum_{j!=i} (1/d_ij) * d(d_ij)/d(x_i)
    # d(d_ij)/d(x_i) = (x_i - x_j)/d_ij
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2 = sum_{j!=i} (x_j - x_i)/d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Compute energy efficiently
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()


def _simulated_annealing(points, start_time, deadline, max_iter=50000):
    """Simulated annealing on the sphere with multi-point moves."""
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    current_points = points.copy()
    current_energy = best_energy
    
    T = 0.02
    T_min = 1e-10
    cooling = 0.9997
    
    rng = np.random.RandomState(12345)
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Occasionally do multi-point moves for better exploration
        if rng.rand() < 0.1:
            # Move 2-3 random points
            num_moves = 2 + (rng.rand() < 0.3)
            indices = rng.choice(N, size=num_moves, replace=False)
            new_points = current_points.copy()
            for idx in indices:
                rand_vec = rng.randn(3)
                tangent = rand_vec - np.dot(rand_vec, current_points[idx]) * current_points[idx]
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-15:
                    continue
                tangent = tangent / tangent_norm
                step_size = T * 3.0
                new_point = current_points[idx] + step_size * tangent
                new_point = new_point / np.linalg.norm(new_point)
                new_points[idx] = new_point
        else:
            # Single point move
            idx = rng.randint(N)
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, current_points[idx]) * current_points[idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-15:
                continue
            tangent = tangent / tangent_norm
            step_size = T * 2.0
            new_point = current_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points = current_points.copy()
            new_points[idx] = new_point
        
        new_energy = _log_energy(new_points)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / max(T, T_min)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T = max(T * cooling, T_min)
    
    return best_points, best_energy


def _gradient_descent_sphere(points, max_iter=10000, lr=0.01, start_time=None, deadline=None):
    """Gradient descent on the sphere with adaptive learning rate."""
    current_points = points.copy()
    best_points = points.copy()
    best_energy = _log_energy(current_points)
    current_energy = best_energy
    
    for i in range(max_iter):
        if start_time is not None and time.time() > deadline - 2.0:
            break
        
        # Compute gradient efficiently
        n = len(current_points)
        dots = current_points @ current_points.T
        dists_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dists_sq, 1.0)
        dists_sq = np.maximum(dists_sq, 1e-30)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current_points
        
        grads = weighted_sum - current_points * sum_weights[:, np.newaxis]
        
        # Project to tangent space
        dot_products = np.sum(grads * current_points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current_points
        
        # Adaptive learning rate with line search
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break
        
        # Try several step sizes
        accepted = False
        for attempt in range(5):
            step = lr * (0.5 ** attempt)
            new_points = current_points - step * grads_tangent
            new_points = _project_to_sphere(new_points)
            
            new_energy = _log_energy(new_points)
            if new_energy < current_energy:
                current_points = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current_points.copy()
                lr = min(lr * 1.1, 0.2)
                accepted = True
                break
        
        if not accepted:
            lr = lr * 0.5
        
        if lr < 1e-12:
            break
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
        _random_uniform_sphere(N, seed=7),
    ]
    
    # Add perturbed Fibonacci
    rng = np.random.RandomState(42)
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.01
        inits.append(_project_to_sphere(perturbed))
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS-B optimization from multiple starts (limit to top initializations)
    # Sort initializations by energy and optimize the best ones
    init_energies = [_log_energy(init) for init in inits]
    init_order = np.argsort(init_energies)
    
    num_lbfgs = min(5, len(inits))  # Only optimize top 5 starts
    for rank in range(num_lbfgs):
        if time.time() > deadline - 40.0:
            break
        
        idx = init_order[rank]
        x0 = inits[idx].flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 30}
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient descent refinement
    if time.time() < deadline - 25.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, max_iter=30000, lr=0.01, 
            start_time=start_time, deadline=deadline
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()
    
    # Phase 3: Simulated annealing for basin exploration
    if time.time() < deadline - 15.0:
        sa_points, sa_energy = _simulated_annealing(
            best_points, start_time, deadline, max_iter=50000
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if time.time() < deadline - 8.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 5: Final gradient descent micro-polish
    if time.time() < deadline - 3.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, max_iter=5000, lr=0.001,
            start_time=start_time, deadline=deadline
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
