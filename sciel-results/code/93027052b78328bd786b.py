"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy_fast(pts):
    """Faster vectorized energy computation."""
    n = len(pts)
    # Compute all pairwise squared distances
    sq_dists = np.sum(pts**2, axis=1)[:, np.newaxis] + np.sum(pts**2, axis=1)[np.newaxis, :] - 2 * pts @ pts.T
    np.maximum(sq_dists, 0, out=sq_dists)  # numerical stability
    dists = np.sqrt(sq_dists)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only sum upper triangle
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dists[iu]))

def _tangent_grad_fast(pts):
    """Compute gradient of log energy projected onto tangent space, vectorized."""
    n = len(pts)
    # Compute all pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    # Compute distances
    dists = np.sqrt(np.sum(diff**2, axis=2))  # (n, n)
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    # Gradient contribution: for each i, sum over j != i of (x_i - x_j) / |x_i - x_j|^2
    # = sum_j (x_i - x_j) / dists[i,j]^2
    inv_dists_sq = 1.0 / (dists**2)
    inv_dists_sq[np.diag_indices(n)] = 0.0
    # grad[i] = sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    # Project to tangent space: grad_tangent = grad - (grad·x)x
    dots = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dots * pts
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

def _gradient_descent(pts, lr=0.01, max_iter=200, time_limit=None):
    """Simple gradient descent with momentum on the sphere."""
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy_fast(pts)
    
    velocity = np.zeros_like(pts)
    momentum = 0.9
    
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        grad = _tangent_grad_fast(pts)
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

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    best_pts = None
    best_energy = 1e18
    
    # Try multiple initializations and refinements
    initializations = []
    
    # Fibonacci spiral
    fib_pts = _initial_fibonacci()
    initializations.append(fib_pts)
    
    # Random initializations with different seeds
    for seed in [42, 123, 456]:
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
        if time.time() > deadline - 5:
            break
        refined_pts, refined_energy = _gradient_descent(pts, lr=0.01, max_iter=100, time_limit=deadline - 5)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined_pts
    
    # If we still have time, try more aggressive optimization on the best
    if best_pts is not None and time.time() < deadline - 10:
        further_pts, further_energy = _gradient_descent(best_pts, lr=0.001, max_iter=200, time_limit=deadline - 5)
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
                    rng = np.random.RandomState(42)
                    best_pts[j] += 1e-8 * rng.randn(3)
                    best_pts = _project_to_sphere(best_pts)
                    break
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
