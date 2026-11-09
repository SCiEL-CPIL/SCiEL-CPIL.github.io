import numpy as np
import time

def _rotation_matrix(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz

def _densify_path(path):
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start, path, end])
    densified = [full_path[0]]
    for i in range(1, len(full_path)):
        prev = densified[-1]
        curr = full_path[i]
        pos_diff = np.linalg.norm(curr[:3] - prev[:3])
        angle_diffs = np.abs(curr[3:6] - prev[3:6])
        max_angle_diff = np.max(angle_diffs)
        n_steps = max(1, int(np.ceil(max(pos_diff / 0.05, max_angle_diff / 2.0))))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            densified.append(prev + t * (curr - prev))
    return np.array(densified)

def _evaluate_path(path, grid_x, grid_y, grid_z, deadline=None):
    densified = _densify_path(path)
    nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)
    survival = np.ones((nx, ny, nz), dtype=bool)
    px = grid_x[:, None, None]
    py = grid_y[None, :, None]
    pz = grid_z[None, None, :]
    for step in range(len(densified)):
        x, y, z, yaw, pitch, roll = densified[step]
        R = _rotation_matrix(roll, pitch, yaw)
        new_x = R[0,0]*px + R[0,1]*py + R[0,2]*pz + x
        new_y = R[1,0]*px + R[1,1]*py + R[1,2]*pz + y
        new_z = R[2,0]*px + R[2,1]*py + R[2,2]*pz + z
        in_corr = (
            ((new_x <= 0) & (new_y >= 0) & (new_y <= 1) & (new_z >= 0) & (new_z <= 1)) |
            ((new_x >= 0) & (new_x <= 1) & (new_y >= 0) & (new_y <= 5) & (new_z >= 0) & (new_z <= 1)) |
            ((new_x >= 0) & (new_x <= 1) & (new_y >= 5) & (new_y <= 6) & (new_z >= 0))
        )
        survival &= in_corr
        if not np.any(survival):
            break
        if deadline is not None and time.time() > deadline:
            return 0
    n_blocks = 0
    for i in range(nx - 3):
        for j in range(ny - 3):
            for k in range(nz - 3):
                if np.all(survival[i:i+4, j:j+4, k:k+4]):
                    n_blocks += 1
    return n_blocks

def moving_sofa_3d_path() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 590  # 590s to leave margin
    
    path = np.array([
        [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
        # ... (same as current)
    ], dtype=np.float64)
    
    grid_x = np.linspace(-4, 0, 81)
    grid_y = np.linspace(0, 1, 21)
    grid_z = np.linspace(0, 1, 21)
    
    best_path = path.copy()
    best_score = _evaluate_path(best_path, grid_x, grid_y, grid_z, deadline)
    
    # Local search
    np.random.seed(42)
    step_sizes = [0.05, 0.02, 0.01, 0.005]
    for ss in step_sizes:
        while time.time() < deadline:
            pose_idx = np.random.randint(len(best_path))
            dim = np.random.randint(6)
            delta = np.random.choice([-ss, ss])
            trial = best_path.copy()
            trial[pose_idx, dim] += delta
            score = _evaluate_path(trial, grid_x, grid_y, grid_z, deadline)
            if score > best_score:
                best_path = trial
                best_score = score
    
    return best_path
