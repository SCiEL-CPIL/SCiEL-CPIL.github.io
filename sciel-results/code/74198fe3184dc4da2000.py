"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def _log_energy_and_grad(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|) and its gradient.
    
    Energy = sum_{i<j} -log(d_ij) where d_ij = |xi - xj|
    Gradient w.r.t. xi: sum_{j != i} (xi - xj) / d_ij^2
    """
    # Compute pairwise distances
    # d_ij = |xi - xj|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(d, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient: for each point i, grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    # diff[i,j] = xi - xj, so grad_i = sum_j diff[i,j] / d[i,j]^2
    # Note: d[i,i] = 0, but we set it to 1.0 so diff[i,i]/d[i,i]^2 = 0/1 = 0
    inv_d2 = 1.0 / (d ** 2)
    inv_d2[np.diag_indices(N)] = 0.0  # ensure diagonal is zero
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _rnn_step(pts, lr, max_iter=100, tol=1e-10):
    """Riemannian gradient descent step with adaptive learning rate."""
    best_pts = pts.copy()
    best_energy = _log_energy_and_grad(pts)[0]
    
    for it in range(max_iter):
        energy, grad = _log_energy_and_grad(pts)
        
        # Check convergence
        if it > 0 and abs(energy - best_energy) < tol:
            break
        
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Adaptive learning rate
        # Use a simple heuristic: if energy increases, reduce lr
        step = lr * grad_tan
        new_pts = _project_to_sphere(pts + step)
        new_energy = _log_energy_and_grad(new_pts)[0]
        
        if new_energy < best_energy:
            best_pts = new_pts.copy()
            best_energy = new_energy
            pts = new_pts
            # Slightly increase lr if we made good progress
            if it % 10 == 0:
                lr *= 1.05
        else:
            # Reduce learning rate
            lr *= 0.7
            if lr < 1e-8:
                break
    
    return best_pts, best_energy


def _velocity_verlet(pts, dt=0.01, n_steps=5000, damping=0.999):
    """Velocity-Verlet integration for the logarithmic repulsion force field.
    
    Force on point i: F_i = sum_{j != i} (xi - xj) / |xi - xj|^2 (repulsive)
    We integrate with damping to dissipate energy.
    """
    pts = pts.copy()
    vel = np.zeros_like(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy_and_grad(pts)[0]
    
    def _force(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, 1.0)
        inv_d2 = 1.0 / (d ** 2)
        inv_d2[np.diag_indices(len(p))] = 0.0
        return np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    f = _force(pts)
    
    for step in range(n_steps):
        # Velocity-Verlet
        pts = _project_to_sphere(pts + vel * dt + 0.5 * f * dt * dt)
        f_new = _force(pts)
        vel = (vel + 0.5 * (f + f_new) * dt) * damping
        f = f_new
        
        # Project velocity to tangent space
        vel -= np.sum(vel * pts, axis=1, keepdims=True) * pts
        
        # Check energy periodically
        if step % 100 == 0:
            energy = _log_energy_and_grad(pts)[0]
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
    
    return best_pts, best_energy


def _fibonacci_sphere(n):
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - start_time)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy_and_grad(pts)[0]
    
    # Phase 1: Velocity-Verlet integration
    if time_left() > 30:
        pts, energy = _velocity_verlet(pts, dt=0.005, n_steps=20000, damping=0.9995)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Riemannian gradient descent polish
    if time_left() > 15:
        lr = 0.01
        pts, energy = _rnn_step(best_pts, lr, max_iter=200, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: More velocity-Verlet with smaller steps
    if time_left() > 20:
        pts, energy = _velocity_verlet(best_pts, dt=0.001, n_steps=30000, damping=0.9999)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 4: Final Riemannian polish
    if time_left() > 10:
        lr = 0.001
        pts, energy = _rnn_step(best_pts, lr, max_iter=500, tol=1e-14)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Try a few random restarts if time permits
    if time_left() > 40:
        for seed in [42, 123, 999]:
            if time_left() < 35:
                break
            rpts = _random_sphere(N, seed=seed)
            rpts, r_energy = _velocity_verlet(rpts, dt=0.005, n_steps=15000, damping=0.9995)
            if time_left() > 20:
                rpts, r_energy = _rnn_step(rpts, lr=0.01, max_iter=150, tol=1e-12)
            if r_energy < best_energy:
                best_energy = r_energy
                best_pts = rpts.copy()
    
    # Final high-precision polish
    if time_left() > 5:
        lr = 0.0005
        pts, energy = _rnn_step(best_pts, lr, max_iter=1000, tol=1e-15)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Ensure points are on unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    norms = np.linalg.norm(best_pts, axis=1)
    if not np.all(np.isfinite(best_pts)) or np.any(norms < 1e-9):
        # Fallback to Fibonacci
        return _fibonacci_sphere(N)
    
    return best_pts
