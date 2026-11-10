import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Pre-calculate indices for min distance calculation
    triu_indices = np.triu_indices_from(np.zeros((N, N)), k=1)
    
    def generate_hex_lattice(n_points):
        r_est = np.sqrt(np.pi / (2 * np.sqrt(3) * n_points))
        spacing = 2 * r_est
        
        k = int(1.0 / spacing) + 2
        
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3) / 2
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dists)]
        
        if len(points) >= n_points:
            return points[:n_points], r_est
        else:
            # Fallback
            pos = np.random.uniform(-1, 1, (n_points, 2))
            r_pos = np.linalg.norm(pos, axis=1)
            pos = pos / np.max(r_pos) * 0.9
            return pos, 0.01

    pos, r_init = generate_hex_lattice(N)
    pos = pos.astype(np.float64)
    vel = np.zeros_like(pos)
    
    r_target = r_init
    best_r = 0.0
    best_pos = pos.copy()
    
    dt = 0.02
    damping = 0.9
    force_const = 1.0
    boundary_const = 1.0
    steps_per_iter = 50
    
    # Main Loop
    while time.time() < deadline:
        # Compute Forces
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        dist_mat_safe = np.where(dist_mat > 1e-8, dist_mat, 1e-8)
        
        overlap = np.maximum(0, 2 * r_target - dist_mat)
        forces = np.sum(overlap[:, :, np.newaxis] * diff / dist_mat_safe[:, :, np.newaxis], axis=1) * force_const
        
        r_pos = np.linalg.norm(pos, axis=1)
        r_pos_safe = np.where(r_pos > 1e-8, r_pos, 1e-8)
        b_overlap = np.maximum(0, r_pos - (1 - r_target))
        forces -= b_overlap[:, np.newaxis] * pos / r_pos_safe[:, np.newaxis] * boundary_const
        
        # Update
        vel = (vel + forces * dt) * damping
        pos += vel * dt
        
        # Check Stability and Update r_target
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        # Calculate actual radius
        min_dist = np.min(dist_mat[triu_indices])
        min_bound = np.min(1 - r_pos)
        current_r = min(min_dist / 2, min_bound)
        
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
            
        if max_force < 1e-4:
            r_target += 1e-5
            vel *= 0.5
        else:
            # If unstable, do nothing special, damping handles it
            pass

        if not np.all(np.isfinite(pos)):
            pos = best_pos.copy()
            vel = np.zeros_like(pos)
            r_target = best_r

    return best_pos
