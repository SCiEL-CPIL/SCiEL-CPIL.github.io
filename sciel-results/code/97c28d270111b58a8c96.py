"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(points)
    # Compute pairwise distances
    # points is (N, 3)
    # dist[i,j] = |points[i] - points[j]|
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # We need sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _gradient(points):
    """Compute gradient of log energy w.r.t. each point on sphere.
    E = sum_{i<j} log(1/|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            grad[i] += d / (dist * dist)
    return grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_descent(points, max_iters=5000, tol=1e-12, verbose=False):
    """Projected gradient descent on sphere."""
    points = points.copy()
    n = len(points)
    E = _log_energy(points)
    
    # Adaptive step size
    lr = 0.01
    min_lr = 1e-8
    
    for it in range(max_iters):
        g = _gradient(points)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        # Normalized gradient direction
        g_dir = g / (g_norm + 1e-15)
        
        # Try step
        new_points = points - lr * g
        new_points = _project_to_sphere(new_points)
        new_E = _log_energy(new_points)
        
        if new_E < E:
            E = new_E
            points = new_points
            # Increase step size slightly if improvement
            lr = min(lr * 1.1, 0.1)
        else:
            # Decrease step size
            lr *= 0.5
            if lr < min_lr:
                break
    
    return points, E

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    E = _log_energy(points)
    best_points = points.copy()
    best_E = E
    
    # Run gradient descent from Fibonacci start
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        points, E = _gradient_descent(points, max_iters=3000)
        if E < best_E:
            best_points = points.copy()
            best_E = E
    
    # Try a few random restarts if time permits
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Random start
        rng_points = _random_sphere(N, seed=restart * 42 + 7)
        # Quick gradient descent
        points, E = _gradient_descent(rng_points, max_iters=2000)
        if E < best_E:
            best_points = points.copy()
            best_E = E
    
    # Final polish with smaller steps
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        points, E = _gradient_descent(best_points, max_iters=1000, tol=1e-14)
        if E < best_E:
            best_points = points.copy()
            best_E = E
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[j] = best_points[j] + 1e-8 * best_points[i]
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
