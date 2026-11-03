# EVOLVE-BLOCK-START
import numpy as np


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_forces(pts, power):
    """
    Calculate repulsive forces on points on the sphere.
    Potential V = sum_{i<j} 1/|p_i - p_j|^power
    Force F_i = -grad V = sum_{j!=i} power * (p_i - p_j) / |p_i - p_j|^(power+2)
    """
    N = len(pts)
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # dist2[i, j] = |pts[i] - pts[j]|^2
    dist2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    
    # Force magnitude factor: power / dist^(power+2) = power * dist^-(power+2)
    # dist = sqrt(dist2), so dist^-(power+2) = dist2^(-(power+2)/2)
    # f_mag[i,j] = power * dist2[i,j]^(-(power+2)/2)
    exponent = -(power + 2) / 2.0
    f_mag = power * np.power(dist2, exponent)
    
    # Zero out self-forces
    np.fill_diagonal(f_mag, 0.0)
    
    # Total force on each point i: sum_j f_mag[i,j] * diff[i,j]
    forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
    
    # Project forces onto the tangent plane of the sphere at each point
    # F_tangent = F - (F . p) p
    dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
    forces -= dot_fp * pts
    
    return forces


def construct():
    """
    Minimize a repulsive potential on the unit sphere using symplectic integration.
    Uses a higher power to encourage uniform distribution (Tammes-like).
    """
    N = 50
    
    # 1. Initialization: Fibonacci Sphere (good starting point for symmetry)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters
    # Higher power leads to steeper potential, better separation
    power = 10 
    max_steps = 20000
    
    # Pre-compute initial forces
    forces = _compute_forces(pts, power)
    
    for step in range(max_steps):
        # Progress 0 to 1
        t = step / max_steps
        
        # Adaptive step size: start large, decay
        # dt should be small enough for stability: dt ~ sqrt(m/k) ~ 1/sqrt(force_magnitude)
        # We use a heuristic cooling schedule
        base_dt = 0.05
        dt = base_dt * (1.0 - t) ** 3 + 0.0001
        
        # Damping: increase damping as we approach the end to settle
        damping = 0.995 - 0.01 * t
        
        # --- Velocity Verlet Update ---
        
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # Project vel_half to tangent plane at current pts
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        # x_new = x + dt * v_half
        pts_new = pts + dt * vel_half
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Recalculate forces at new position
        forces_new = _compute_forces(pts_new, power)
        
        # v_new = v_half + 0.5 * dt * F_new
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Apply damping
        vel_new *= damping
        
        # Project vel_new to tangent plane at new position
        dot_vp_new = np.sum(vel_new * pts_new, axis=1, keepdims=True)
        vel_new -= dot_vp_new * pts_new
        
        # Update state
        pts = pts_new
        vel = vel_new
        forces = forces_new
        
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
