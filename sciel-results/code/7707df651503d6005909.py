"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    N = 300
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_pos = None

    def compute_score(pos):
        if not np.all(np.isfinite(pos)):
            return -1.0
        dists = np.linalg.norm(pos, axis=1)
        if np.any(dists > 1.0 + 1e-9):
            return -1.0
        
        diffs = pos[:, None, :] - pos[None, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist = np.sqrt(np.min(dists_sq))
        
        min_boundary = 1.0 - np.max(dists)
        
        if min_boundary < -1e-9:
            return -1.0
            
        r = min(min_dist / 2.0, min_boundary)
        return r

    def get_hex_init():
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        points = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            while len(points) < N:
                s *= 0.95
                points = []
                k = int(1.0 / s) + 2
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * s
                        y = j * s * np.sqrt(3.0) / 2.0
                        r = np.hypot(x, y)
                        if r <= 1.0:
                            points.append([x, y])
                points = np.array(points)
        
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]

    def run_langevin(pos0, steps=5000, dt=0.002, gamma=1.0, k_rep=50.0, d_cut=0.3, k_wall=500.0):
        pos = pos0.copy()
        vel = np.zeros_like(pos)
        m = 1.0
        T0 = 0.1
        T1 = 0.001
        
        for step in range(steps):
            T = T0 + (T1 - T0) * (step / steps)
            
            diffs = pos[:, None, :] - pos[None, :, :]
            dists_sq = np.sum(diffs**2, axis=2)
            dists = np.sqrt(dists_sq)
            
            mask = (dists < d_cut) & (dists > 1e-8)
            
            forces = np.zeros_like(pos)
            if np.any(mask):
                d3 = dists_sq * dists
                d3_safe = np.where(mask, d3, 1.0)
                F_mat = k_rep * diffs / d3_safe[:, :, None]
                F_mat[~mask] = 0.0
                forces += np.sum(F_mat, axis=1)
            
            r_norms = np.linalg.norm(pos, axis=1)
            wall_mask = r_norms > 1.0
            if np.any(wall_mask):
                penetration = r_norms[wall_mask] - 1.0
                F_wall = -k_wall * penetration[:, None] * pos[wall_mask] / r_norms[wall_mask][:, None]
                forces[wall_mask] += F_wall
            
            noise = np.random.normal(0, np.sqrt(2 * gamma * T * dt / m), size=pos.shape)
            vel = vel + (forces / m - gamma / m * vel) * dt + noise
            pos = pos + vel * dt
            
            r_norms = np.linalg.norm(pos, axis=1)
            too_far = r_norms > 1.5
            if np.any(too_far):
                pos[too_far] = pos[too_far] / r_norms[too_far][:, None] * 1.5
                vel[too_far] = 0.0
                
        return pos

    for restart in range(10):
        if time.time() > deadline:
            break
            
        pos0 = get_hex_init()
        pos0 += np.random.normal(0, 0.01, size=pos0.shape)
        
        pos = run_langevin(pos0)
        
        r_norms = np.linalg.norm(pos, axis=1)
        outside = r_norms > 1.0
        if np.any(outside):
            pos[outside] = pos[outside] / r_norms[outside][:, None]
            
        score = compute_score(pos)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()

    if best_pos is None:
        pos0 = get_hex_init()
        r_norms = np.linalg.norm(pos0, axis=1)
        outside = r_norms > 1.0
        if np.any(outside):
            pos0[outside] = pos0[outside] / r_norms[outside][:, None]
        best_pos = pos0

    return best_pos
# EVOLVE-BLOCK-END
