# EVOLVE-BLOCK-START
import numpy as np

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_min_dist(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    min_dist = np.sqrt(np.min(dist2, axis=1))
    return np.min(min_dist)

def construct():
    """
    Minimize a repulsive potential on the unit sphere using Langevin dynamics 
    with simulated annealing. Multiple restarts are performed.
    """
    N = 50
    best_min_dist = 0.0
    best_pts = None
    
    power = 10
    steps_per_restart = 30000
    restarts = 5
    dt = 0.002
    gamma = 1.0
    
    T_start = 0.2
    T_end = 0.001
    
    for r in range(restarts):
        # Initialization: Fibonacci Sphere with random rotation
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r_rad = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r_rad * np.cos(phi), r_rad * np.sin(phi), z])
        
        # Random rotation
        q = np.random.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, zq = q
        R = np.array([
            [1 - 2*(y*y + zq*zq), 2*(x*y - w*zq), 2*(x*zq + w*y)],
            [2*(x*y + w*zq), 1 - 2*(x*x + zq*zq), 2*(y*zq - w*x)],
            [2*(x*zq - w*y), 2*(y*zq + w*x), 1 - 2*(x*x + y*y)]
        ])
        pts = pts @ R.T
        
        vel = np.zeros((N, 3))
        
        # Initial forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist2, 1.0)
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = power * inv_dist ** (power + 2)
        np.fill_diagonal(f_mag, 0.0)
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        for step in range(steps_per_restart):
            t = step / steps_per_restart
            T = T_start * (T_end / T_start) ** t
            
            # Langevin step
            noise_scale = np.sqrt(2 * gamma * T * dt)
            noise = noise_scale * np.random.normal(size=(N, 3))
            
            # Damped update
            # v = (v + dt * F) * (1 - gamma * dt) + noise
            # Ensure stability: if gamma*dt is large, this might oscillate.
            # dt=0.002, gamma=1.0 -> 0.002. Safe.
            vel = (vel + dt * forces) * (1.0 - gamma * dt) + noise
            
            # Project velocity to tangent
            dot_vp = np.sum(vel * pts, axis=1, keepdims=True)
            vel -= dot_vp * pts
            
            pts = pts + dt * vel
            pts = _normalize(pts)
            
            # Recompute forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist2, 1.0)
            inv_dist = 1.0 / np.sqrt(dist2)
            f_mag = power * inv_dist ** (power + 2)
            np.fill_diagonal(f_mag, 0.0)
            forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
            dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
            forces -= dot_fp * pts
            
        # End of Langevin
        # Local polish: Gradient Descent
        for step in range(1000):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist2, 1.0)
            inv_dist = 1.0 / np.sqrt(dist2)
            f_mag = power * inv_dist ** (power + 2)
            np.fill_diagonal(f_mag, 0.0)
            forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
            dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
            forces -= dot_fp * pts
            
            # Adaptive step size for GD
            # Norm of forces
            f_norm = np.linalg.norm(forces)
            if f_norm < 1e-6:
                break
            step_size = 0.01 / (1.0 + f_norm)
            
            pts = pts + step_size * forces
            pts = _normalize(pts)
            
        current_min_dist = _compute_min_dist(pts)
        
        if current_min_dist > best_min_dist:
            best_min_dist = current_min_dist
            best_pts = pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
