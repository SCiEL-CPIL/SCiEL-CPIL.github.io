"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total

def _energy_and_grad(X):
    """Compute energy and gradient of log-energy on the sphere."""
    n = X.shape[0]
    energy = 0.0
    grad = np.zeros_like(X)
    for i in range(n):
        for j in range(i+1, n):
            diff = X[i] - X[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return float('inf'), grad
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j|^2 = -(x_i-x_j)/|x_i-x_j|^3
            # But we want gradient of sum log(1/d) = -sum log(d)
            # d/dx_i (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
            # So gradient contribution to x_i is -(xi-xj)/d^2, to x_j is +(xi-xj)/d^2
            term = -diff / (d * d)
            grad[i] += term
            grad[j] -= term
            energy += math.log(1.0 / d)
    return energy, grad

def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms

def _project_grad(X, grad):
    """Project gradient onto tangent space of sphere at each point."""
    return grad - np.sum(grad * X, axis=1, keepdims=True) * X

def _exponential_map(X, V, step):
    """Exponential map on sphere: move along geodesic."""
    normV = np.linalg.norm(V, axis=1, keepdims=True)
    normV[normV < 1e-15] = 1e-15
    sin_term = np.sin(step * normV) / normV
    cos_term = np.cos(step * normV)
    return X * cos_term + V * sin_term

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_sphere(n):
    """Generate points using Halton sequences mapped to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    points = []
    for i in range(1, n+1):
        u = halton(i, 2)
        v = halton(i, 3)
        theta = 2.0 * np.pi * u
        z = 1.0 - 2.0 * v
        r = np.sqrt(1.0 - z * z)
        points.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(points)

def _farthest_point_init(n, seed=42):
    """Initialize using farthest-point sampling for better spread."""
    rng = np.random.RandomState(seed)
    # Start with two antipodal points
    X = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    while len(X) < n:
        # Find farthest point from current set
        best_dist = -1
        best_point = None
        for _ in range(200):
            candidate = rng.randn(3)
            candidate /= np.linalg.norm(candidate)
            # Compute min distance to existing points
            dists = [np.linalg.norm(candidate - p) for p in X]
            min_dist = min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = candidate
        X = np.vstack([X, best_point])
    return X

def _riemannian_descent(X, max_iter=500, tol=1e-10, time_limit=100.0):
    """Riemannian gradient descent with adaptive step size."""
    start_time = time.time()
    X = _normalize(X.copy())
    best_X = X.copy()
    best_energy = _log_energy(X)
    
    step = 0.01
    prev_energy = best_energy
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        energy, grad = _energy_and_grad(X)
        if not math.isfinite(energy):
            break
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Project gradient to tangent space
        pgrad = _project_grad(X, grad)
        grad_norm = np.linalg.norm(pgrad)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        step = min(0.1, 0.01 / (grad_norm + 1e-6))
        
        # Try the step
        new_X = _exponential_map(X, pgrad, step)
        new_X = _normalize(new_X)
        new_energy = _log_energy(new_X)
        
        if math.isfinite(new_energy) and new_energy < energy:
            X = new_X
            # Increase step if improvement was good
            if new_energy < energy - 0.01:
                step *= 1.2
        else:
            # Decrease step
            step *= 0.5
            if step < 1e-8:
                break
        
        if abs(energy - prev_energy) < 1e-12:
            break
        prev_energy = energy
    
    return best_X, best_energy

def _polish(X, time_limit=20.0):
    """Final high-precision polish with very small steps."""
    start_time = time.time()
    X = _normalize(X.copy())
    best_X = X.copy()
    best_energy = _log_energy(X)
    step = 0.001
    
    for it in range(1000):
        if time.time() - start_time > time_limit:
            break
        energy, grad = _energy_and_grad(X)
        if not math.isfinite(energy):
            break
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        pgrad = _project_grad(X, grad)
        grad_norm = np.linalg.norm(pgrad)
        if grad_norm < 1e-12:
            break
        
        new_X = _exponential_map(X, pgrad, step)
        new_X = _normalize(new_X)
        new_energy = _log_energy(new_X)
        
        if math.isfinite(new_energy) and new_energy < energy:
            X = new_X
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    return best_X, best_energy

def construct():
    global N
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Try multiple initialization strategies
    inits = [
        ('fibonacci', _fibonacci_sphere(N)),
        ('halton', _halton_sphere(N)),
        ('farthest', _farthest_point_init(N, seed=42)),
        ('farthest2', _farthest_point_init(N, seed=123)),
    ]
    
    best_X = None
    best_energy = float('inf')
    
    for name, X in inits:
        if time.time() - start_time > deadline - 30:
            break
        X, energy = _riemannian_descent(X, max_iter=300, tol=1e-10, time_limit=30.0)
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
    
    # Final polish on the best result
    if time.time() - start_time < deadline - 10:
        best_X, best_energy = _polish(best_X, time_limit=30.0)
    
    return _normalize(best_X)
# EVOLVE-BLOCK-END
