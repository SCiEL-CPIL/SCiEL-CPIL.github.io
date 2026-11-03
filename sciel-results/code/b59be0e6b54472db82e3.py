# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Fast approximation using a dense grid
    n_theta = 180
    n_phi = 360
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    Y = np.zeros((n_theta, n_phi, 3))
    Y[:, :, 0] = np.sin(TH) * np.cos(PH)
    Y[:, :, 1] = np.sin(TH) * np.sin(PH)
    Y[:, :, 2] = np.cos(TH)
    Y = Y.reshape(-1, 3)
    
    # Compute max dot product (min angular distance) for each grid point
    dots = Y @ pts.T
    min_dots = np.max(dots, axis=1)
    max_angular = np.arccos(np.clip(min_dots, -1, 1))
    return np.degrees(np.max(max_angular))


def _compute_forces(pts, power):
    """Compute repulsive forces for inverse power law potential."""
    N = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist2, 1.0)
    
    inv_dist = 1.0 / np.sqrt(dist2)
    # Force magnitude for V = 1/r^power is -dV/dr = power * r^-(power+1)
    # Direction is diff (from j to i)
    f_mag = inv_dist ** (power + 1)
    np.fill_diagonal(f_mag, 0.0)
    
    forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
    return forces


def construct():
    """
    Minimize a repulsive potential on the unit sphere using Velocity Verlet integration.
    Uses annealing of the power law to escape local minima and refine the configuration.
    """
    N = 50
    
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters
    # We will run multiple stages with different powers
    # Stage 1: Lower power (4) for global exploration
    # Stage 2: Medium power (10) for intermediate refinement
    # Stage 3: High power (20) for final Tammes-like packing
    
    stages = [
        {'power': 4, 'steps': 20000, 'dt_start': 0.01, 'dt_end': 0.001, 'damp_start': 0.99, 'damp_end': 0.90},
        {'power': 10, 'steps': 20000, 'dt_start': 0.005, 'dt_end': 0.0005, 'damp_start': 0.99, 'damp_end': 0.90},
        {'power': 20, 'steps': 30000, 'dt_start': 0.002, 'dt_end': 0.0001, 'damp_start': 0.99, 'damp_end': 0.95}
    ]
    
    for stage in stages:
        power = stage['power']
        max_steps = stage['steps']
        
        # Initial forces for Verlet
        forces = _compute_forces(pts, power)
        
        for step in range(max_steps):
            t = step / max_steps
            
            # Annealing parameters
            dt = stage['dt_start'] * (1.0 - t) + stage['dt_end']
            damping = stage['damp_start'] - (stage['damp_start'] - stage['damp_end']) * t
            
            # Velocity Verlet Step
            # 1. Update velocity half-step
            vel_half = vel + 0.5 * dt * forces
            
            # 2. Update position
            pts_new = pts + dt * vel_half
            
            # 3. Normalize position to unit sphere
            pts_new = _normalize(pts_new)
            
            # 4. Compute new forces
            forces_new = _compute_forces(pts_new, power)
            
            # 5. Update velocity full-step
            vel_new = vel_half + 0.5 * dt * forces_new
            
            # 6. Apply damping to velocity
            vel_new = vel_new * damping
            
            # 7. Project velocity onto tangent plane
            dot_vp = np.sum(vel_new * pts_new, axis=1, keepdims=True)
            vel_new -= dot_vp * pts_new
            
            # Update state
            pts = pts_new
            vel = vel_new
            forces = forces_new
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
