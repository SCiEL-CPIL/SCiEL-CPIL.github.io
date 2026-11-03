"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i (before projection) for the term log(1/d) = -log(d):
      d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    Summed over all j != i.
    """
    n = pts.shape[0]
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
    np.fill_diagonal(dist, 1.0)  # avoid division by zero
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # term[i,j] = diff[i,j] / dist[i,j]^2
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist_sq = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _local_optimize(pts, maxiter=5000):
    """Run L-BFGS-B optimization starting from pts."""
    def objective(v):
        p = _project_to_sphere(v.reshape(N, 3))
        e, g = _energy_and_grad(p)
        # Chain rule for projection: gradient in v-space
        # For simplicity, just use the gradient on the sphere and project
        # Actually, for L-BFGS-B we need gradient in the unconstrained space
        # Let's use a different approach: parameterize on the sphere directly
        return e, g.flatten()
    
    # Use a simpler approach: just minimize on the sphere using projected gradient
    # Or use scipy with constraints
    # Let's use a custom approach with projected gradient descent
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Use L-BFGS-B with reparameterization
    # Actually, let's use a simpler iterative approach
    
    lr = 0.01
    for it in range(maxiter):
        e, g = _energy_and_grad(best_pts)
        # Project gradient onto tangent space
        g_tangent = g - np.sum(g * best_pts, axis=1, keepdims=True) * best_pts
        # Update
        new_pts = best_pts - lr * g_tangent
        new_pts = _project_to_sphere(new_pts)
        new_e = _energy_and_grad(new_pts)[0]
        if new_e < best_energy:
            best_pts = new_pts
            best_energy = new_e
            # Adaptive learning rate
            lr = min(lr * 1.01, 0.1)
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_pts, best_energy


def _gradient_descent_with_momentum(pts, steps=10000, lr=0.01, momentum=0.9):
    """Gradient descent with momentum on the sphere."""
    best_pts = pts.copy()
    best_energy = _energy_and_grad(best_pts)[0]
    v = np.zeros_like(best_pts)
    
    for it in range(steps):
        e, g = _energy_and_grad(best_pts)
        g_tangent = g - np.sum(g * best_pts, axis=1, keepdims=True) * best_pts
        v = momentum * v - lr * g_tangent
        new_pts = _project_to_sphere(best_pts + v)
        new_e = _energy_and_grad(new_pts)[0]
        if new_e < best_energy:
            best_pts = new_pts
            best_energy = new_e
        else:
            v *= 0.5
    
    return best_pts, best_energy


def _anneal_and_optimize(pts, time_budget):
    """Gradient-flow annealing followed by local optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Annealing phase: gradual cooling
    n_anneal_steps = 5000
    for it in range(n_anneal_steps):
        if time.time() - start_time > time_budget * 0.4:
            break
        e, g = _energy_and_grad(best_pts)
        g_tangent = g - np.sum(g * best_pts, axis=1, keepdims=True) * best_pts
        # Temperature schedule
        temp = 0.01 * (1.0 - it / n_anneal_steps) ** 2
        noise = np.random.normal(0, temp, size=best_pts.shape)
        noise = noise - np.sum(noise * best_pts, axis=1, keepdims=True) * best_pts
        lr = 0.005
        new_pts = _project_to_sphere(best_pts - lr * g_tangent + noise)
        new_e = _energy_and_grad(new_pts)[0]
        if new_e < best_energy:
            best_pts = new_pts
            best_energy = new_e
    
    # Local optimization phase
    if time.time() - start_time < time_budget * 0.8:
        best_pts, best_energy = _local_optimize(best_pts, maxiter=3000)
    
    # Final polish with tiny perturbations
    if time.time() - start_time < time_budget * 0.95:
        for _ in range(20):
            if time.time() - start_time > time_budget * 0.95:
                break
            perturb = np.random.normal(0, 1e-4, size=best_pts.shape)
            perturb = perturb - np.sum(perturb * best_pts, axis=1, keepdims=True) * best_pts
            perturbed = _project_to_sphere(best_pts + perturb)
            p_e, _ = _energy_and_grad(perturbed)
            if p_e < best_energy:
                best_pts = perturbed
                best_energy = p_e
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral()
    
    # Run annealing and optimization
    best_pts, best_energy = _anneal_and_optimize(pts, time_budget=135)
    
    # Check deadline
    if time.time() > deadline:
        return _project_to_sphere(best_pts)
    
    # Try a few random restarts if time permits
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        random_pts = _project_to_sphere(np.random.normal(size=(N, 3)))
        r_pts, r_energy = _anneal_and_optimize(random_pts, time_budget=25)
        if r_energy < best_energy:
            best_pts = r_pts
            best_energy = r_energy
    
    return _project_to_sphere(best_pts)
# EVOLVE-BLOCK-END
