"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Deadline guard: return best result before 150s wall-clock
_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


def _energy_and_grad(points):
    """Compute energy and gradient on the sphere.
    
    Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient for point i: -sum_{j!=i} (xi - xj) / |xi-xj|^2  (then project to tangent)
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n,n,3)
    sq_dists = np.sum(diff * diff, axis=2)  # (n,n) squared distances
    dists = np.sqrt(np.maximum(sq_dists, 1e-30))  # (n,n)
    
    # inv_d2[i,j] = 1/|xi-xj|^2 for i != j, 0 for i==j
    inv_d2 = np.zeros_like(sq_dists)
    mask = sq_dists > 1e-24
    inv_d2[mask] = 1.0 / sq_dists[mask]
    
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space
    grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))
    
    return energy, grad_tangent


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _halton_sphere(n, rng):
    """Generate n points using Halton sequence for better coverage."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(n):
        u1 = halton(i + 1, 2)
        u2 = halton(i + 1, 3)
        z = 1.0 - 2.0 * u1
        theta = 2.0 * np.pi * u2
        r = np.sqrt(max(0, 1.0 - z * z))
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, 1e-15)


def _adam_tangent(points, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=5000):
    """Adam optimizer on the tangent space of the sphere with adaptive LR."""
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    t = 0
    
    best_energy = _energy(points)
    best_points = points.copy()
    current_lr = lr
    patience = 0
    max_patience = 200
    
    for it in range(max_iters):
        if _remaining_time() < 10.0:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate decay
        if patience > max_patience:
            current_lr *= 0.5
            patience = 0
            if current_lr < 1e-7:
                break
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = -current_lr * m_hat / (np.sqrt(v_hat) + eps)
        update_tangent = update - np.sum(update * points, axis=1, keepdims=True) * points
        
        points = points + update_tangent
        points = _project_to_sphere(points)
    
    return best_points, best_energy


def _lbfgs_tangent(points, max_iters=3000, tol=1e-12):
    """L-BFGS optimization using scipy with retraction approach."""
    from scipy.optimize import minimize
    
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'maxls': 30})
    
    if result.success or result.fun < best_energy:
        best_points = _project_to_sphere(result.x.reshape(n, 3))
        best_energy = float(result.fun)
    
    return best_points, best_energy


def _lbfgs_tight(points, max_iters=8000, tol=1e-15):
    """Very tight L-BFGS polish for final refinement."""
    from scipy.optimize import minimize
    
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'maxls': 50})
    
    if result.success or result.fun < best_energy:
        best_points = _project_to_sphere(result.x.reshape(n, 3))
        best_energy = float(result.fun)
    
    return best_points, best_energy


def _structured_perturb(points, rng, scale=0.05):
    """Apply structured perturbation: rotate subsets of points slightly."""
    pts = points.copy()
    n = len(pts)
    # Apply small random rotations to random subsets
    for _ in range(3):
        subset_size = rng.integers(5, 15)
        idx = rng.choice(n, size=subset_size, replace=False)
        # Random rotation axis
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-scale, scale)
        # Rodrigues rotation
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
        pts[idx] = pts[idx] @ R.T
    return _project_to_sphere(pts)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Start with multiple diverse initializations
    best_points = _fibonacci_sphere(N)
    best_energy = _energy(best_points)
    
    # Also try Halton as initial
    halton_pts = _halton_sphere(N, rng)
    halton_energy = _energy(halton_pts)
    if halton_energy < best_energy:
        best_energy = halton_energy
        best_points = halton_pts.copy()
    
    # Multi-start Adam optimization with diverse initializations
    n_restarts = 12
    
    for restart in range(n_restarts):
        if _remaining_time() < 20.0:
            break
        
        if restart == 0:
            init_points = best_points.copy()
            lr = 0.01
            max_it = 4000
        elif restart == 1:
            init_points = _random_sphere(N, rng)
            lr = 0.01
            max_it = 3000
        elif restart == 2:
            init_points = _halton_sphere(N, rng)
            lr = 0.01
            max_it = 3000
        elif restart <= 4:
            # Small structured perturbation
            init_points = _structured_perturb(best_points, rng, scale=0.03)
            lr = 0.008
            max_it = 3000
        elif restart <= 7:
            # Medium perturbation
            noise = rng.standard_normal((N, 3)) * 0.08
            init_points = _project_to_sphere(best_points + noise)
            lr = 0.006
            max_it = 2500
        else:
            # Larger perturbation
            noise = rng.standard_normal((N, 3)) * 0.2
            init_points = _project_to_sphere(best_points + noise)
            lr = 0.005
            max_it = 2000
        
        pts, energy = _adam_tangent(
            init_points,
            lr=lr,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            max_iters=max_it
        )
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # L-BFGS polish stages
    if _remaining_time() > 20.0:
        pts, energy = _lbfgs_tangent(best_points, max_iters=3000, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    if _remaining_time() > 15.0:
        pts, energy = _lbfgs_tight(best_points, max_iters=8000, tol=1e-15)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Final ultra-tight polish if time allows
    if _remaining_time() > 8.0:
        pts, energy = _lbfgs_tight(best_points, max_iters=10000, tol=1e-16)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    dists = np.linalg.norm(best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = dists[iu].min()
    if min_dist < 1e-9:
        best_points = _fibonacci_sphere(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
