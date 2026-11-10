# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view


def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in degrees."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz


def _in_corridor_vec(x, y, z):
    """Vectorized corridor check. Returns bool array."""
    r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return r1 | r2 | r3


def _densify_path(full_path):
    """Densify path so each step moves at most 0.05 Euclidean and changes each angle by at most 2 degrees."""
    densified = [full_path[0]]
    for i in range(1, len(full_path)):
        p0 = full_path[i - 1]
        p1 = full_path[i]
        pos_dist = np.linalg.norm(p1[:3] - p0[:3])
        ang_diffs = np.abs(p1[3:] - p0[3:])
        n_steps = max(1, int(np.ceil(pos_dist / 0.05)), int(np.ceil(np.max(ang_diffs) / 2.0)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            interp = p0 + t * (p1 - p0)
            densified.append(interp)
    return np.array(densified)


def _build_grid():
    """Build the grid of points once."""
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return points


def _evaluate_path_fast(path, points=None):
    """
    Fast vectorized evaluation of the path.
    Returns the surviving volume (count of 4x4x4 blocks * 0.05^3).
    """
    if points is None:
        points = _build_grid()
    
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start, path, end])
    densified = _densify_path(full_path)
    
    N = len(points)
    alive = np.ones(N, dtype=bool)
    
    # Check at first pose
    p0 = densified[0]
    R0 = _rotation_matrix(p0[3], p0[4], p0[5])
    placed0 = points @ R0.T + p0[:3]
    alive &= _in_corridor_vec(placed0[:, 0], placed0[:, 1], placed0[:, 2])
    
    # Process each step
    for step in range(1, len(densified)):
        if not np.any(alive):
            break
        prev = densified[step - 1]
        curr = densified[step]
        
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dz = curr[2] - prev[2]
        dyaw = curr[3] - prev[3]
        dpitch = curr[4] - prev[4]
        droll = curr[5] - prev[5]
        
        # x move
        if abs(dx) > 1e-12:
            p_x = prev.copy()
            p_x[0] = curr[0]
            R_x = _rotation_matrix(p_x[3], p_x[4], p_x[5])
            placed_x = points[alive] @ R_x.T + p_x[:3]
            new_alive = _in_corridor_vec(placed_x[:, 0], placed_x[:, 1], placed_x[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
            
        # y move
        if abs(dy) > 1e-12:
            p_y = prev.copy()
            p_y[0] = curr[0]
            p_y[1] = curr[1]
            R_y = _rotation_matrix(p_y[3], p_y[4], p_y[5])
            placed_y = points[alive] @ R_y.T + p_y[:3]
            new_alive = _in_corridor_vec(placed_y[:, 0], placed_y[:, 1], placed_y[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
            
        # z move
        if abs(dz) > 1e-12:
            p_z = prev.copy()
            p_z[0] = curr[0]
            p_z[1] = curr[1]
            p_z[2] = curr[2]
            R_z = _rotation_matrix(p_z[3], p_z[4], p_z[5])
            placed_z = points[alive] @ R_z.T + p_z[:3]
            new_alive = _in_corridor_vec(placed_z[:, 0], placed_z[:, 1], placed_z[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # Rotation - check if single axis is dominant
        # If multiple axes change, we check the final pose (approximation)
        # If single axis, we check the arc by sampling
        active_axes = []
        if abs(dyaw) > 1e-12: active_axes.append('yaw')
        if abs(dpitch) > 1e-12: active_axes.append('pitch')
        if abs(droll) > 1e-12: active_axes.append('roll')
        
        if len(active_axes) == 1:
            axis = active_axes[0]
            p_rot = prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            
            if axis == 'yaw':
                p_rot[3] = curr[3]
                n_samples = max(2, int(np.ceil(abs(dyaw) / 2.0)))
                for s in range(n_samples + 1):
                    t = s / n_samples
                    yaw_s = prev[3] + t * dyaw
                    R_s = _rotation_matrix(yaw_s, p_rot[4], p_rot[5])
                    placed_s = points[alive] @ R_s.T + p_rot[:3]
                    new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                    alive_idx = np.where(alive)[0]
                    alive[alive_idx[~new_alive]] = False
            elif axis == 'pitch':
                p_rot[4] = curr[4]
                n_samples = max(2, int(np.ceil(abs(dpitch) / 2.0)))
                for s in range(n_samples + 1):
                    t = s / n_samples
                    pitch_s = prev[4] + t * dpitch
                    R_s = _rotation_matrix(p_rot[3], pitch_s, p_rot[5])
                    placed_s = points[alive] @ R_s.T + p_rot[:3]
                    new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                    alive_idx = np.where(alive)[0]
                    alive[alive_idx[~new_alive]] = False
            elif axis == 'roll':
                p_rot[5] = curr[5]
                n_samples = max(2, int(np.ceil(abs(droll) / 2.0)))
                for s in range(n_samples + 1):
                    t = s / n_samples
                    roll_s = prev[5] + t * droll
                    R_s = _rotation_matrix(p_rot[3], p_rot[4], roll_s)
                    placed_s = points[alive] @ R_s.T + p_rot[:3]
                    new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                    alive_idx = np.where(alive)[0]
                    alive[alive_idx[~new_alive]] = False
        else:
            # Multiple axes: check final pose
            R_rot = _rotation_matrix(curr[3], curr[4], curr[5])
            placed_rot = points[alive] @ R_rot.T + curr[:3]
            new_alive = _in_corridor_vec(placed_rot[:, 0], placed_rot[:, 1], placed_rot[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
    
    # Count 4x4x4 blocks
    alive_3d = alive.reshape(81, 21, 21)
    blocks = sliding_window_view(alive_3d, (4, 4, 4))
    all_survive = np.all(blocks, axis=(3, 4, 5))
    count = np.sum(all_survive)
    return count * (0.05 ** 3)


def _generate_decoupled_path():
    """
    Generate a path that strictly decouples moves and rotations.
    This ensures that every densified step involves at most one angle change.
    """
    poses = []
    
    # 1. Move from start (-3, 0, 0) to entrance of region 1 (-0.5, 0.5, 0.5)
    # Start pose is fixed by evaluator: (-3, 0, 0, 0, 0, 0)
    # We start our path from there.
    # Move X to -0.5
    poses.append([-0.5, 0.0, 0.0, 0, 0, 0])
    # Move Y to 0.5
    poses.append([-0.5, 0.5, 0.0, 0, 0, 0])
    # Move Z to 0.5
    poses.append([-0.5, 0.5, 0.5, 0, 0, 0])
    
    # 2. Move through region 1 to x=0.5 (center of region 2 entrance)
    poses.append([0.5, 0.5, 0.5, 0, 0, 0])
    
    # 3. Rotate Yaw 0 -> 90 to align with Y-axis corridor
    # We do this in small steps? No, the densifier handles it.
    # But we must ensure position is fixed during rotation.
    poses.append([0.5, 0.5, 0.5, 90, 0, 0])
    
    # 4. Move along Y from 0.5 to 5.5 (center of region 3 entrance)
    poses.append([0.5, 5.5, 0.5, 90, 0, 0])
    
    # 5. Rotate Pitch 0 -> 90 to align with Z-axis corridor
    poses.append([0.5, 5.5, 0.5, 90, 90, 0])
    
    # 6. Move along Z from 0.5 to 7.0 (end z)
    # End pose is fixed: (0, 6, 7, 90, 90, 90)
    # We need to get to x=0, y=6 as well.
    # Let's move to (0.5, 5.5, 7.0) first
    poses.append([0.5, 5.5, 7.0, 90, 90, 0])
    
    # 7. Rotate Roll 0 -> 90
    poses.append([0.5, 5.5, 7.0, 90, 90, 90])
    
    # 8. Move to final position (0, 6, 7)
    # Move X 0.5 -> 0
    poses.append([0.0, 5.5, 7.0, 90, 90, 90])
    # Move Y 5.5 -> 6
    poses.append([0.0, 6.0, 7.0, 90, 90, 90])
    
    return np.array(poses)


def _optimize_path(path, points, time_limit, rng):
    """
    Optimize the path by perturbing positions and angles.
    We use a decoupled approach: perturb positions of 'move' poses and angles of 'rotate' poses?
    Actually, we can just perturb any pose, but we must be careful not to break the decoupling if we want to keep it.
    However, the evaluator accepts multi-axis steps (likely), so we can just hill climb the whole path.
    """
    best_path = path.copy()
    best_score = _evaluate_path_fast(best_path, points)
    
    base_pos_sigma = 0.05
    base_ang_sigma = 1.0
    min_pos_sigma = 0.005
    min_ang_sigma = 0.1
    max_iter = 1000
    
    for iter_num in range(max_iter):
        if time.time() > time_limit:
            break
        
        progress = iter_num / max_iter
        pos_sigma = base_pos_sigma * (1.0 - progress) + min_pos_sigma * progress
        ang_sigma = base_ang_sigma * (1.0 - progress) + min_ang_sigma * progress
        
        idx = rng.integers(0, len(best_path))
        trial_pose = best_path[idx].copy()
        
        # Perturb
        trial_pose[0] += rng.normal(0, pos_sigma)
        trial_pose[1] += rng.normal(0, pos_sigma)
        trial_pose[2] += rng.normal(0, pos_sigma)
        trial_pose[3] += rng.normal(0, ang_sigma)
        trial_pose[4] += rng.normal(0, ang_sigma)
        trial_pose[5] += rng.normal(0, ang_sigma)
        
        test_path = best_path.copy()
        test_path[idx] = trial_pose
        trial_score = _evaluate_path_fast(test_path, points)
        
        if trial_score > best_score:
            best_path = test_path
            best_score = trial_score
            
    return best_path, best_score


def moving_sofa_3d_path() -> np.ndarray:
    """
    The 3D moving sofa problem.
    Uses decoupled path generation followed by local optimization.
    """
    t_start = time.time()
    deadline = t_start + 570.0  # Leave 30s margin
    
    points = _build_grid()
    rng = np.random.default_rng(42)
    
    # Generate initial decoupled path
    initial_path = _generate_decoupled_path()
    
    # Evaluate initial
    current_path = initial_path.copy()
    current_score = _evaluate_path_fast(current_path, points)
    
    # Optimization Phase 1: Coarse search
    time_limit_1 = t_start + 200
    current_path, current_score = _optimize_path(current_path, points, time_limit_1, rng)
    
    # Optimization Phase 2: Fine search
    time_limit_2 = deadline - 10
    current_path, current_score = _optimize_path(current_path, points, time_limit_2, rng)
    
    return current_path


# EVOLVE-BLOCK-END
