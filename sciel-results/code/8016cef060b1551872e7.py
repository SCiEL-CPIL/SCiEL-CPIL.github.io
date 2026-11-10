"""
Strategy: Molecular Dynamics (Langevin dynamics) with cooling on the sphere.
Unlike gradient descent which follows the steepest path, MD uses stochastic
force-based updates with thermal noise to escape local minima.
"""

import numpy as np
import time
import math

N = 150

def _energy(points):
    """Calculate logarithmic energy. Returns float."""
    # points: (N, 3)
    n = len(points)
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero for i==j
    np.fill_diagonal(dist, 1.0)
    # Energy is sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # We sum over all i != j and divide by 2
    log_dists = np.log(dist)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    total_energy = -0.5 * np.sum(log_dists[mask])
    return total_energy

def _forces(points):
    """
    Compute forces (negative gradient of energy) on each point.
    Energy E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    Gradient of log(d_ij) w.r.t x_i is (x_i - x_j) / d_ij^2
    So -dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0) # Avoid div by 0, will mask later
    
    # Force term: (x_i - x_j) / d_ij^2
    # We want sum_j (x_i - x_j) / d_ij^2
    # diff[i,j] = x_i - x_j
    # factor[i,j] = 1 / d_ij^2
    inv_dist_sq = np.zeros_like(dist)
    mask = ~np.eye(n, dtype=bool)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # forces[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return forces

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    # MD Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    damping = 0.9  # Velocity damping
    dt = 0.01      # Time step
    cooling_rate = 0.995
    
    velocity = np.zeros_like(points)
    
    iter_count = 0
    while time.time() < deadline:
        # Compute forces
        F = _forces(points)
        
        # Langevin update:
        # v = damping * v + dt * F + sqrt(2*T*dt) * noise
        noise = np.random.normal(0, 1, size=points.shape) * np.sqrt(2 * T * dt)
        velocity = damping * velocity + dt * F + noise
        
        # Update positions
        points = points + dt * velocity
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Compute energy
        current_energy = _energy(points)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            # Reheat slightly to escape local minima if we have time
            if time.time() < deadline - 20:
                T = 0.005
                velocity *= 0.1 # Damp velocity on reheat
            else:
                T = T_min
        
        iter_count += 1
        
        # Periodic check for timeout
        if time.time() > deadline:
            break
            
    return best_points
