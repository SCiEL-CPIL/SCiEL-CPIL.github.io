"""Optimized logarithmic energy minimizer for 150 points on the unit sphere using Langevin Dynamics."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _compute_forces(points):
    """
    Compute the force (negative gradient of energy) on each point.
    Force on i due to j is -d/dx_i log(1/|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    Wait, Energy E = sum log(1/d) = - sum log(d).
    dE/dx_i = - sum_j d(log d_ij)/dx_i = - sum_j (1/d_ij) * (x_i - x_j)/d_ij = - sum_j (x_i - x_j)/d_ij^2.
    Force F_i = - dE/dx_i = sum_j (x_i - x_j)/d_ij^2.
    This force pushes points apart.
    """
    n = len(points)
    forces = np.zeros_like(points)
    # Vectorized distance calculation
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-15] = 1e-15
    
    # Force term: (x_i - x_j) / dist^2
    # We need to sum over j for each i.
    # factor[i, j] = 1 / dist_sq[i, j]
    factor = 1.0 / dist_sq
    # Zero out self-interaction
    np.fill_diagonal(factor, 0.0)
    
    # forces[i] = sum_j diff[i, j] * factor[i, j]
    forces = np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    return forces

def _project_to_sphere(points):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy(points):
    """Compute logarithmic energy."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total

def _local_refinement(points, max_iter=1000):
    """
    Simple gradient descent refinement to polish the result after annealing.
    """
    pts = points.copy()
    step = 0.01
    for _ in range(max_iter):
        forces = _compute_forces(pts)
        # Project forces to tangent space
        for i in range(N):
            proj = np.dot(forces[i], pts[i]) * pts[i]
            forces[i] -= proj
        
        new_pts = pts + step * forces
        new_pts = _project_to_sphere(new_pts)
        
        e_old = _energy(pts)
        e_new = _energy(new_pts)
        
        if e_new < e_old:
            pts = new_pts
            step *= 1.05 # Slightly increase step if improving
            if step > 0.1: step = 0.1
        else:
            step *= 0.8
            if step < 1e-6:
                break
    return pts

def construct():
    # Initialize
    points = _initial_fibonacci()
    
    # Langevin Dynamics / Simulated Annealing
    # Parameters
    T_initial = 0.1
    T_final = 1e-4
    cooling_rate = 0.995
    dt = 0.01
    gamma = 0.9 # Friction coefficient
    
    velocities = np.zeros_like(points)
    rng = np.random.default_rng(42)
    
    T = T_initial
    iters = 0
    max_iters = 5000
    
    while T > T_final and iters < max_iters:
        forces = _compute_forces(points)
        
        # Langevin update
        # v = gamma * v + force * dt + noise
        noise = rng.normal(0, math.sqrt(2 * T * dt), size=points.shape)
        velocities = gamma * velocities + forces * dt + noise
        
        # Update positions
        points = points + velocities * dt
        
        # Project to sphere
        points = _project_to_sphere(points)
        
        # Cool down
        T *= cooling_rate
        iters += 1
        
        # Optional: Occasionally run a few steps of local refinement to help escape shallow traps?
        # No, keep it pure Langevin for the "new approach" claim, but we can refine at the end.
        
    # Final local refinement
    points = _local_refinement(points)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                # Perturb
                points[j] += 1e-6 * rng.randn(3)
                points = _project_to_sphere(points)
                
    return points
# EVOLVE-BLOCK-END
