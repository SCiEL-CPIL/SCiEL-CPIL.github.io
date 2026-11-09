"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e18
            total += -math.log(d)
    return total

def _log_energy_fast(pts):
    """Faster vectorized energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only sum upper triangle
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dists[iu]))

def _tangent_grad(pts):
    """Compute gradient of log energy projected onto tangent space."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dxi of -log|xi-xj| = (xi-xj)/|xi-xj|^2
            grad[i] += diff / (d * d)
    # Project to tangent space: grad_tangent = grad - (grad·x)x
    for i in range(n):
        dot = np.dot(grad[i], pts[i])
        grad[i] -= dot * pts[i]
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(seed=42):
    """Random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _lbfgs_step(pts, lr=0.01, max_iter=1000):
    """Simple gradient descent with momentum on the sphere."""
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy_fast(pts)
    
    velocity = np.zeros_like(pts)
    momentum = 0.9
    
    for it in range(max_iter):
        grad = _tangent_grad(pts)
        velocity = momentum * velocity - lr * grad
        new_pts = pts + velocity
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy_fast(new_pts)
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
        else:
            lr *= 0.5
            velocity = 0.0
    
    return pts, current_energy

def _local_refine(pts, time_limit=None):
    """Refine points using gradient descent."""
    start = time.time()
    pts, energy = _lbfgs_step(pts, lr=0.01, max_iter=500)
    if time_limit and (time.time() - start) > time_limit * 0.5:
        return pts, energy
    # Second pass with smaller steps
    pts, energy = _lbfgs_step(pts, lr=0.001, max_iter=500)
    return pts, energy

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    best_pts = None
    best_energy = 1e18
    
    # Try multiple initializations and refinements
    initializations = []
    
    # Fibonacci spiral
    fib_pts = _initial_fibonacci()
    initializations.append(fib_pts)
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        rand_pts = _initial_random(seed)
        initializations.append(rand_pts)
    
    # Sort by initial energy
    init_energies = []
    for pts in initializations:
        e = _log_energy_fast(pts)
        init_energies.append((e, pts))
    init_energies.sort(key=lambda x: x[0])
    
    # Refine the best initializations
    for e_init, pts in init_energies:
        if time.time() > deadline - 10:
            break
        refined_pts, refined_energy = _local_refine(pts)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined_pts
    
    # If we still have time, try more aggressive optimization
    if best_pts is not None and time.time() < deadline - 20:
        # Try further refinement
        further_pts, further_energy = _lbfgs_step(best_pts, lr=0.0001, max_iter=1000)
        if further_energy < best_energy:
            best_energy = further_energy
            best_pts = further_pts
    
    # Final projection and validation
    if best_pts is not None:
        best_pts = _project_to_sphere(best_pts)
        # Ensure no points are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-9:
                    # Nudge apart
                    best_pts[j] += 1e-8 * np.random.randn(3)
                    best_pts = _project_to_sphere(best_pts)
                    break
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
