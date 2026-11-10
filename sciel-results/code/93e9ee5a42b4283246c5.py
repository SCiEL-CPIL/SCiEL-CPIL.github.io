"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 150

# EVOLVE-BLOCK-START

def _log_energy(points):
    """Compute sum_{i<j} log(1/|xi-xj|) using math.fsum for accuracy."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points.
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-18:
                # Avoid division by zero; add small penalty
                d2 = 1e-18
            grad[i] += diff / d2
    return grad


def _energy_and_grad_flat(flat):
    """Energy and gradient for flat vector input (for L-BFGS-B)."""
    points = flat.reshape(N, 3)
    e = _log_energy(points)
    if e == float('inf'):
        return e, np.zeros(3 * N)
    g = _log_energy_grad(points)
    return e, g.ravel()


def _project_to_sphere(points):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return points / norms


def _energy_sphere(theta, phi):
    """Energy from spherical coordinates (theta: polar, phi: azimuthal)."""
    sin_t = np.sin(theta)
    points = np.column_stack([
        sin_t * np.cos(phi),
        sin_t * np.sin(phi),
        np.cos(theta)
    ])
    return _log_energy(points)


def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _local_search(points, time_budget, seed=0):
    """Run L-BFGS-B local optimization from given starting points."""
    deadline = time.time() + time_budget
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    flat = points.ravel()
    
    # Use L-BFGS-B with bounds to keep points in reasonable range
    # Actually, we want to optimize on the sphere. Let's use a different approach.
    # We'll optimize in R^3 and then project, but that's not smooth.
    # Better: use the fact that gradient on sphere is the projection of the Euclidean gradient.
    
    # For L-BFGS-B, we need a smooth objective. Let's work in R^3 and add a penalty
    # for being off the sphere, or better, just optimize and project at the end.
    # Actually, the cleanest is to optimize with the constraint |xi|=1.
    # But L-BFGS-B doesn't handle equality constraints well.
    
    # Alternative: use the spherical coordinate approach.
    # But that has singularities at poles.
    
    # Let's just use L-BFGS-B in R^3 and periodically project.
    # Or better: use scipy.optimize.minimize with method='L-BFGS-B' and
    # a penalty for deviation from unit sphere.
    
    def obj_with_penalty(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        penalty = np.sum((norms - 1.0) ** 2) * 1e6
        e = _log_energy(pts)
        if e == float('inf'):
            return 1e10
        return e + penalty
    
    def grad_with_penalty(x):
        pts = x.reshape(N, 3)
        g = _log_energy_grad(pts)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Gradient of penalty: 2*1e6*(norms-1)*pts/norms
        pen_grad = 2.0 * 1e6 * (norms - 1.0) * pts / norms
        return (g + pen_grad).ravel()
    
    res = minimize(
        obj_with_penalty,
        flat,
        jac=grad_with_penalty,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    
    if time.time() > deadline:
        return best_points, best_energy
    
    if res.success or res.fun < best_energy:
        new_points = _project_to_sphere(res.x.reshape(N, 3))
        new_energy = _log_energy(new_points)
        if new_energy < best_energy and new_energy != float('inf'):
            best_points = new_points
            best_energy = new_energy
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    # Try a few random restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate random starting points
        seed = r * 12345 + 678
        rand_points = _random_sphere(N, seed)
        
        # Run local search with time budget
        time_budget = min(20.0, deadline - time.time() - 5.0)
        if time_budget < 5.0:
            break
        
        local_points, local_energy = _local_search(rand_points, time_budget, seed=seed)
        
        if local_energy < best_energy and local_energy != float('inf'):
            best_points = local_points.copy()
            best_energy = local_energy
    
    # Also try local search from the Fibonacci start
    if time.time() < deadline - 10.0:
        time_budget = min(25.0, deadline - time.time() - 5.0)
        if time_budget > 5.0:
            local_points, local_energy = _local_search(best_points, time_budget, seed=42)
            if local_energy < best_energy and local_energy != float('inf'):
                best_points = local_points.copy()
                best_energy = local_energy
    
    # Final check: ensure validity
    final_points = _project_to_sphere(best_points)
    final_energy = _log_energy(final_points)
    
    if final_energy == float('inf'):
        # Fallback to Fibonacci
        final_points = _fibonacci_sphere(N)
    
    return final_points

# EVOLVE-BLOCK-END
