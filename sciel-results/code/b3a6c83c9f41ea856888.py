"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # Using broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid diagonal (zero distances)
    np.fill_diagonal(dist, 1.0)
    # Energy is sum of log(1/d) for i<j
    # log(1/d) = -log(d)
    # We need upper triangle only
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _energy_and_grad(points):
    """Compute energy and gradient. Gradient is the force that decreases energy.
    
    For energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist_sq)
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j!=i} diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _velocity_verlet_step(points, velocities, dt):
    """One velocity-Verlet step with projection onto sphere."""
    n = len(points)
    
    # Compute forces (negative gradient of energy)
    _, grad = _energy_and_grad(points)
    forces = -grad  # force = -gradient
    
    # Update velocities: v_new = v + 0.5 * dt * F
    velocities = velocities + 0.5 * dt * forces
    
    # Update positions: x_new = x + dt * v_new
    points_new = points + dt * velocities
    
    # Project back to sphere
    points_new = _project_to_sphere(points_new)
    
    # Compute new forces
    _, grad_new = _energy_and_grad(points_new)
    forces_new = -grad_new
    
    # Update velocities again: v_new = v + 0.5 * dt * F_new
    velocities = velocities + 0.5 * dt * forces_new
    
    return points_new, velocities


def _damped_dynamics(points, max_time=100.0):
    """Run damped dynamics to minimize energy."""
    n = len(points)
    current_energy = _energy(points)
    
    # Adaptive timestep
    dt = 0.01
    velocities = np.zeros_like(points)
    
    # Velocity-Verlet with damping
    damping = 0.99
    
    max_steps = 5000
    best_points = points.copy()
    best_energy = current_energy
    
    for step in range(max_steps):
        # Check time
        if step % 100 == 0 and time.time() > max_time:
            break
        
        # Compute forces
        _, grad = _energy_and_grad(points)
        forces = -grad
        
        # Damped velocity update
        velocities = damping * velocities + dt * forces
        
        # Update positions
        points_new = points + dt * velocities
        points_new = _project_to_sphere(points_new)
        
        # Compute new energy
        new_energy = _energy(points_new)
        
        if new_energy < current_energy:
            points = points_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
            # Try to increase dt slightly
            dt = min(dt * 1.05, 0.1)
        else:
            # Reject step, reduce dt
            dt *= 0.7
            velocities *= 0.5
        
        if dt < 1e-6:
            break
    
    return best_points, best_energy


def _lbfgs_refine(points, max_iter=1000):
    """Refine using L-BFGS-B from scipy."""
    from scipy.optimize import minimize
    
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        e, g = _energy_and_grad(pts)
        # Gradient with respect to x (before projection)
        # Use chain rule: grad_x = grad_pts * J where J is Jacobian of projection
        # For simplicity, just use the gradient on the sphere and project
        return e, g.flatten()
    
    # Flatten points
    x0 = points.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-12}
    )
    
    pts = _project_to_sphere(result.x.reshape(-1, 3))
    return pts, _energy(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    best_points = None
    best_energy = float('inf')
    
    # Multiple random restarts
    n_restarts = 10
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Generate initial points
        if restart == 0:
            points = _fibonacci_sphere(N)
        else:
            points = _random_sphere(N, seed=rng.integers(0, 2**32))
        
        # Phase 1: Damped dynamics
        try:
            pts, e = _damped_dynamics(points, max_time=min(30.0, deadline - time.time() - 5.0))
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            continue
        
        # Phase 2: L-BFGS refinement if time permits
        if time.time() < deadline - 10.0 and best_points is not None:
            try:
                pts, e = _lbfgs_refine(best_points, max_iter=500)
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
        
        # If we found a good solution, try to improve with more dynamics
        if time.time() < deadline - 5.0 and best_points is not None:
            try:
                pts, e = _damped_dynamics(best_points, max_time=min(15.0, deadline - time.time() - 2.0))
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Final L-BFGS polish if time permits
    if time.time() < deadline - 5.0 and best_points is not None:
        try:
            pts, e = _lbfgs_refine(best_points, max_iter=1000)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
    
    # Ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb slightly
                best_points[j] += np.random.normal(0, 1e-6, 3)
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
