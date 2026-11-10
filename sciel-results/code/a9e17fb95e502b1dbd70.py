# EVOLVE-BLOCK-START
import time
import numpy as np

# --- Constants for the corridor and grid ---
GRID_RES = 0.05
X_GRID = np.linspace(-4, 0, 81)
Y_GRID = np.linspace(0, 1, 21)
Z_GRID = np.linspace(0, 1, 21)

GX, GY, GZ = np.meshgrid(X_GRID, Y_GRID, Z_GRID, indexing='ij')
GRID_POINTS = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
N_POINTS = len(GRID_POINTS)

# Precompute corridor check for speed
def in_corridor(points):
    """Vectorized check for corridor membership. Input: (N, 3) array."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    c1 = (x <= 0) & (0 <= y) & (y <= 1) & (0 <= z) & (z <= 1)
    c2 = (0 <= x) & (x <= 1) & (0 <= y) & (y <= 5) & (0 <= z) & (z <= 1)
    c3 = (0 <= x) & (x <= 1) & (5 <= y) & (y <= 6) & (z >= 0)
    return c1 | c2 | c3

def rotation_matrix(yaw, pitch, roll):
    """Compute R = Rx(roll) Ry(pitch) Rz(yaw). Angles in radians."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    return Rx @ Ry @ Rz

def evaluate_path_fast(path):
    """
    Fast evaluation of the path. 
    We approximate the survival condition by checking the poses directly.
    For the volume count, we use a vectorized approach where possible.
    """
    if len(path) == 0:
        return 0.0
        
    start = np.array([-3, 0, 0, 0, 0, 0])
    end = np.array([0, 6, 7, 90, 90, 90])
    
    all_poses = np.vstack([start, path, end])
    
    alive = np.ones(N_POINTS, dtype=bool)
    
    # Check start
    pts = GRID_POINTS + start[:3]
    alive &= in_corridor(pts)
    if not np.any(alive):
        return 0.0

    # Check intermediate poses
    # To be fast, we batch the rotation calculations if possible, but loops are okay for ~100 poses
    for i in range(len(path)):
        pose = path[i]
        yaw, pitch, roll = np.deg2rad(pose[3]), np.deg2rad(pose[4]), np.deg2rad(pose[5])
        R = rotation_matrix(yaw, pitch, roll)
        pts = (R @ GRID_POINTS.T).T + pose[:3]
        alive &= in_corridor(pts)
        if not np.any(alive):
            return 0.0

    # Check end
    yaw, pitch, roll = np.deg2rad(90), np.deg2rad(90), np.deg2rad(90)
    R = rotation_matrix(yaw, pitch, roll)
    pts = (R @ GRID_POINTS.T).T + end[:3]
    alive &= in_corridor(pts)
    
    if not np.any(alive):
        return 0.0
        
    alive_grid = alive.reshape(81, 21, 21)
    
    # Count 4x4x4 blocks
    # Use sliding window sum or just loop since 78*18*18 is small
    count = 0
    # Optimization: If we assume the alive region is contiguous or large, we can just count.
    # But we need exact count of 4x4x4 all-true blocks.
    # Vectorized sum of windows:
    # Sum of alive in 4x4x4 window. If sum == 64, it's a valid block.
    # We can use np.lib.stride_tricks or just a simple loop for reliability and speed in Python.
    # Loop is ~25k iterations, which is fast (<1ms).
    for i in range(78):
        for j in range(18):
            for k in range(18):
                if np.all(alive_grid[i:i+4, j:j+4, k:k+4]):
                    count += 1
                    
    return count * (0.05 ** 3)

def generate_initial_path():
    """
    Generate a path that follows the center of the corridor.
    Corridor segments:
    1. x <= 0, 0<=y<=1, 0<=z<=1. Center: y=0.5, z=0.5. x goes from -3 to 0.
    2. 0<=x<=1, 0<=y<=5, 0<=z<=1. Center: x=0.5, z=0.5. y goes from 0 to 5.
    3. 0<=x<=1, 5<=y<=6, z>=0. Center: x=0.5, y=5.5. z goes from 0 to 7 (end is z=7).
    
    Start: (-3, 0, 0, 0, 0, 0) -> Actually start is (-3,0,0) with 0 rot.
    End: (0, 6, 7, 90, 90, 90)
    
    We need to interpolate between these segments.
    """
    path = []
    
    # Segment 1: x from -3 to 0, y=0.5, z=0.5, rot=0
    # But start is y=0, z=0. So we need to move y and z first?
    # The evaluator prepends (-3,0,0,0,0,0).
    # Let's start at x=-3, y=0.5, z=0.5.
    # Wait, the start pose is fixed by evaluator: (-3,0,0,0,0,0).
    # So the first move from (-3,0,0) to our first point must be valid.
    # Let's define our path points.
    
    # Point 1: Move to center of first corridor section
    # x=-2.9, y=0.5, z=0.5
    # We will generate points along the "spine".
    
    # Spine 1: x in [-3, 0], y=0.5, z=0.5
    # Spine 2: x=0.5, y in [0, 5], z=0.5
    # Spine 3: x=0.5, y=5.5, z in [0, 7]
    
    # We need to connect these.
    # Let's create a list of key points.
    keypoints = [
        [-3.0, 0.5, 0.5, 0, 0, 0], # Start area (evaluator starts at -3,0,0, so we move y,z)
        [-2.0, 0.5, 0.5, 0, 0, 0],
        [-1.0, 0.5, 0.5, 0, 0, 0],
        [0.0, 0.5, 0.5, 0, 0, 0],
        [0.5, 1.0, 0.5, 0, 0, 0], # Turn
        [0.5, 2.0, 0.5, 0, 0, 0],
        [0.5, 3.0, 0.5, 0, 0, 0],
        [0.5, 4.0, 0.5, 0, 0, 0],
        [0.5, 5.0, 0.5, 0, 0, 0],
        [0.5, 5.5, 1.0, 0, 0, 0], # Turn to Z
        [0.5, 5.5, 2.0, 0, 0, 0],
        [0.5, 5.5, 3.0, 0, 0, 0],
        [0.5, 5.5, 4.0, 0, 0, 0],
        [0.5, 5.5, 5.0, 0, 0, 0],
        [0.5, 5.5, 6.0, 0, 0, 0],
        [0.5, 5.5, 7.0, 0, 0, 0],
    ]
    
    # Interpolate rotations to match end pose (0, 6, 7, 90, 90, 90)
    # The end pose is (0, 6, 7, 90, 90, 90).
    # My spine ends at (0.5, 5.5, 7, 0,0,0).
    # I need to move to (0, 6, 7) and rotate to 90,90,90.
    
    # Let's refine the last part.
    keypoints.append([0.0, 6.0, 7.0, 90, 90, 90])
    
    # Now interpolate between keypoints to get ~100 points.
    # We'll just linearly interpolate position and angles.
    path_points = []
    for i in range(len(keypoints)-1):
        p1 = np.array(keypoints[i])
        p2 = np.array(keypoints[i+1])
        # Determine steps based on distance
        dist = np.linalg.norm(p1[:3] - p2[:3]) + np.linalg.norm(p1[3:] - p2[3:])
        steps = max(1, int(dist / 0.5)) # ~0.5 units per step
        
        for t in np.linspace(0, 1, steps, endpoint=(i==len(keypoints)-2)):
            p = p1 + t * (p2 - p1)
            if i == 0 and t == 0:
                # Skip start if it's exactly the evaluator's start, or keep it?
                # Evaluator prepends start. If we include it, it's fine.
                # But we should ensure the first move is valid.
                pass
            path_points.append(p)
            
    return np.array(path_points)

def moving_sofa_3d_path() -> np.ndarray:
    start_time = time.time()
    deadline = 590 # 10 sec margin
    
    # 1. Generate initial path
    current_path = generate_initial_path()
    current_score = evaluate_path_fast(current_path)
    
    if current_score == 0:
        # Fallback to a very conservative path if initial fails
        # Just return a path that stays in the center
        return current_path

    # 2. Coordinate Descent Optimization
    # We perturb individual poses or small groups.
    # Since the corridor is narrow, moving away from the center reduces volume.
    # So we want to stay close to the "spine".
    
    # We will use a simple hill climbing with adaptive step size.
    step_size = 0.05
    n_iter = 0
    max_iter = 5000
    
    while n_iter < max_iter:
        if time.time() - start_time > deadline:
            break
        n_iter += 1
        
        # Pick a random pose to perturb
        idx = np.random.randint(0, len(current_path))
        
        # Perturbation
        candidate = current_path.copy()
        # Perturb position
        pos_pert = np.random.normal(0, step_size, 3)
        # Perturb rotation
        rot_pert = np.random.normal(0, step_size * 2, 3) # Angles can be larger
        
        candidate[idx, :3] += pos_pert
        candidate[idx, 3:] += rot_pert
        
        # Ensure we don't jump too far (evaluator interpolates, but let's keep steps reasonable)
        # Actually, the evaluator handles interpolation. We just need the poses to be valid.
        
        new_score = evaluate_path_fast(candidate)
        
        if new_score > current_score:
            current_path = candidate
            current_score = new_score
            step_size = min(step_size * 1.05, 0.5)
        else:
            step_size = max(step_size * 0.95, 0.001)
            
            # If stuck for a while, try a larger jump or reset step size
            if n_iter % 100 == 0:
                step_size = 0.1
                
    # Final check
    if not np.all(np.isfinite(current_path)):
        return generate_initial_path()
        
    return current_path

# EVOLVE-BLOCK-END
