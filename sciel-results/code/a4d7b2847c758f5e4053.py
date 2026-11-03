"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy(points):
    """Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    # Compute all pairwise distances
    # Use the identity |x_i - x_j|^2 = 2 - 2*x_i·x_j for unit vectors
    dots = points @ points.T
    dists_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dists_sq, 1.0)  # avoid log(0) on diagonal
    dists_sq = np.maximum(dists_sq, 1e-30)
    dists = np.sqrt(dists_sq)
    # Upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Use fsum for precision
    terms = np.log(1.0 / d)
    return math.fsum(terms)


def _log_energy_and_grad(points):
    """Compute logarithmic energy and its Euclidean gradient."""
    n = points.shape[0]
    dots = points @ points.T
    dists_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dists_sq, 1.0)
    dists_sq = np.maximum(dists_sq, 1e-30)
    dists = np.sqrt(dists_sq)
    
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    # Gradient: d/dx_i [log(1/|x_i - x_j|)] = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j| 
    # Wait: d/dx_i log(1/|x_i - x_j|) = d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # So the force on point i from point j is -(x_i - x_j)/|x_i - x_j|^2 = (x_j - x_i)/|x_i - x_j|^2
    # This is a repulsive force (points push apart).
    
    # grad_i = sum_{j != i} -(x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # Vectorized computation
    # diff[i,j] = x_i - x_j, shape (n, n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    # dists_sq is (n, n)
    # inv_d2 = 1/dists_sq, with diagonal set to 0
    inv_d2 = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_d2[mask] = 1.0 / dists_sq[mask]
    
    # grad[i] = sum_j (x_j - x_i) / |x_i - x_j|^2 = -sum_j (x_i - x_j) / |x_i - x_j|^2
    # = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _velocity_verlet_step(points, velocities, dt):
    """One velocity-Verlet step with projection to sphere."""
    n = points.shape[0]
    
    # Compute forces at current positions
    _, grad = _log_energy_and_grad(points)
    # Force is -gradient of energy (we want to minimize energy)
    # But wait: grad is dE/dx, so force = -grad for descent
    # Actually for MD, we want the force that drives the system.
    # The potential is E(x), so force = -dE/dx = -grad
    forces = -grad
    
    # Velocity Verlet:
    # x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt^2
    # v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
    
    # But we need to handle the constraint x·x = 1.
    # Simple approach: do the update in R^3, then project to sphere, 
    # and remove radial velocity component.
    
    a = forces  # acceleration = force (unit mass)
    
    # Update positions
    new_points = points + velocities * dt + 0.5 * a * dt * dt
    new_points = _project_to_sphere(new_points)
    
    # Compute new forces
    _, new_grad = _log_energy_and_grad(new_points)
    new_forces = -new_grad
    
    # Update velocities
    new_velocities = velocities + 0.5 * (a + new_forces) * dt
    
    # Remove radial component of velocity (constrain to tangent space)
    # v_tangent = v - (v·x̂)x̂
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    unit_points = new_points / norms
    radial = np.sum(new_velocities * unit_points, axis=1, keepdims=True)
    new_velocities = new_velocities - radial * unit_points
    
    return new_points, new_velocities


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    velocities = np.zeros_like(points)
    
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    # Adaptive timestep
    dt = 0.01
    dt_min = 1e-5
    dt_max = 0.1
    damping = 0.999  # slight damping to help convergence
    
    # Velocity Verlet integration with adaptive dt
    step = 0
    while time.time() < deadline:
        # Try a step
        new_points, new_velocities = _velocity_verlet_step(points, velocities, dt)
        
        # Check for numerical issues
        if not np.all(np.isfinite(new_points)):
            dt *= 0.5
            if dt < dt_min:
                break
            continue
        
        # Compute new energy
        new_energy = _log_energy(new_points)
        
        if not np.isfinite(new_energy):
            dt *= 0.5
            if dt < dt_min:
                break
            continue
        
        # If energy decreased, accept and possibly increase dt
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points.copy()
        
        if new_energy < best_energy or new_energy < _log_energy(points) * 0.999:
            # Energy is decreasing, try to increase dt
            if step % 100 == 0:
                dt = min(dt * 1.05, dt_max)
        else:
            # Energy increased, reduce dt
            dt = max(dt * 0.9, dt_min)
        
        points = new_points
        velocities = new_velocities * damping  # apply damping
        
        step += 1
        
        # If we've been stuck for a while, try a different strategy
        if step % 500 == 0:
            # Check if we're making progress
            current_energy = _log_energy(points)
            if current_energy > best_energy * 1.0001:
                # Reset to best and restart with smaller dt
                points = best_points.copy()
                velocities = np.zeros_like(points)
                dt = min(dt, 0.005)
    
    # Final polish: use gradient descent with backtracking
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        points = best_points.copy()
        current_energy = _log_energy(points)
        
        lr = 0.01
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            
            energy, grad = _log_energy_and_grad(points)
            if not np.isfinite(energy):
                break
            
            # Project gradient to tangent space
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            unit_points = points / norms
            radial = np.sum(grad * unit_points, axis=1, keepdims=True)
            grad_tangent = grad - radial * unit_points
            
            # Backtracking line search
            alpha = lr
            for _ in range(20):
                new_points = points - alpha * grad_tangent
                new_points = _project_to_sphere(new_points)
                new_energy = _log_energy(new_points)
                if new_energy < energy:
                    break
                alpha *= 0.5
            
            points = new_points
            energy = new_energy
            
            if energy < current_energy:
                current_energy = energy
                best_energy = energy
                best_points = points.copy()
            
            # Decrease learning rate slowly
            lr *= 0.999
    
    # Final normalization
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for _ in range(100):
        dots = best_points @ best_points.T
        dists_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(np.maximum(dists_sq, 0.0))
        if np.min(dists) > 1e-9:
            break
        # Nudge points that are too close
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        for idx in range(len(d)):
            if d[idx] < 1e-8:
                i, j = iu[0][idx], iu[1][idx]
                # Small random perturbation
                noise = np.random.randn(3) * 1e-8
                best_points[i] += noise
                best_points[j] -= noise
                best_points[i] /= np.linalg.norm(best_points[i])
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
