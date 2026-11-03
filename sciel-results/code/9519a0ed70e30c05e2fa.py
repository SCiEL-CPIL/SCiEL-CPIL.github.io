# EVOLVE-BLOCK-START
import numpy as np

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_forces(pts, power=6):
    """
    Compute repulsive forces for potential V = sum 1/r^power.
    F_i = -grad_i V = sum_{j!=i} power * r_ij^(-power-1) * (p_i - p_j)
    """
    N = pts.shape[0]
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # dist2[i, j] = |pts[i] - pts[j]|^2
    dist2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    
    # r_ij = sqrt(dist2)
    # f_mag = power * r_ij^(-power-1)
    # We can compute r_ij^(-power-1) as (dist2)^(-(power+1)/2)
    # Let k = (power+1)/2
    k = (power + 1) / 2.0
    f_mag = power * (dist2 ** (-k))
    
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
    """
    N = 50
    
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters
    max_steps = 20000
    power = 6  # Use 1/r^6 potential
    
    # Initial step size and damping
    dt_init = 0.02
    dt_final = 0.0001
    damping_init = 0.99
    damping_final = 0.95
    
    # Precompute forces for initial position
    forces = _compute_forces(pts, power)
    
    for step in range(max_steps):
        # Progress 0 to 1
        t = step / max_steps
        
        # Linear cooling schedule
        dt = dt_init * (1.0 - t) + dt_final * t
        damping = damping_init * (1.0 - t) + damping_final * t
        
        # Velocity Verlet Step 1: Update positions
        # x_new = x + v*dt + 0.5*F*dt^2
        pts_new = pts + vel * dt + 0.5 * forces * dt * dt
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Velocity Verlet Step 2: Compute new forces
        forces_new = _compute_forces(pts_new, power)
        
        # Velocity Verlet Step 3: Update velocities
        # v_new = (v + 0.5*(F_old + F_new)*dt) * damping
        vel_new = vel + 0.5 * (forces + forces_new) * dt
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
