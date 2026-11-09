import numpy as np
import time

def moving_sofa_3d_path() -> np.ndarray:
    """
    3D moving sofa: find a motion path through an L-shaped 3D corridor.
    Returns (k, 6) float array of poses (x, y, z, yaw, pitch, roll) in degrees.
    """
    t_start = time.time()
    deadline = t_start + 575  # 55s safety margin before 630s kill

    # Corridor is closed: {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
    # Grid: x in linspace(-4,0,81), y in linspace(0,1,21), z in linspace(0,1,21)
    # Block: 4x4x4 adjacent grid points (spacings: dx=4/80=0.05, dy=1/20=0.05, dz=1/20=0.05)
    # Volume = 0.05^3 * (number of 4x4x4 blocks that all survive)

    grid_x = np.linspace(-4, 0, 81)
    grid_y = np.linspace(0, 1, 21)
    grid_z = np.linspace(0, 1, 21)
    dx, dy, dz = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0], grid_z[1] - grid_z[0]

    # Build initial path
    def build_path():
        poses = []
        # Phase 1: Move from (-3,0,0) to (0.5,0.5,0.5), yaw 0->90
        n1 = 20
        for i in range(n1 + 1):
            t = i / n1
            poses.append([-3 + 3.5*t, 0 + 0.5*t, 0 + 0.5*t, 90*t, 0, 0])
        # Phase 2: Move along arm2, y: 0.5->5, x=0.5, z=0.5, yaw=90
        n2 = 20
        for i in range(1, n2 + 1):
            t = i / n2
            poses.append([0.5, 0.5 + 4.5*t, 0.5, 90, 0, 0])
        # Phase 3: Corner turn, pitch 0->90, y:5->5.5, z:0.5->1.0
        n3 = 15
        for i in range(1, n3 + 1):
            t = i / n3
            poses.append([0.5, 5.0 + 0.5*t, 0.5 + 0.5*t, 90, 90*t, 0])
        # Phase 4: Move up, z:1->7, x=0.5, y=5.5
        n4 = 20
        for i in range(1, n4 + 1):
            t = i / n4
            poses.append([0.5, 5.5, 1.0 + 6.0*t, 90, 90, 0])
        # Phase 5: Roll 0->90, x:0.5->0, y:5.5->6, z=7
        n5 = 8
        for i in range(1, n5 + 1):
            t = i / n5
            poses.append([0.5 - 0.5*t, 5.5 + 0.5*t, 7.0, 90, 90, 90*t])
        return np.array(poses, dtype=np.float64)

    def rotation_matrix(yaw, pitch, roll):
        """R = Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in degrees."""
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rx @ Ry @ Rz

    def in_corridor(p):
        """Check if point p is in the closed corridor."""
        x, y, z = p
        if x <= 0 and 0 <= y <= 1 and 0 <= z <= 1:
            return True
        if 0 <= x <= 1 and 0 <= y <= 5 and 0 <= z <= 1:
            return True
        if 0 <= x <= 1 and 5 <= y <= 6 and z >= 0:
            return True
        return False

    def point_survives(p, path):
        """Check if a grid point p survives the entire path."""
        # Prepend start pose and append end pose
        # path already has intermediate poses
        # Full sequence: start, path[0], ..., path[-1], end
        full_poses = np.vstack([[-3, 0, 0, 0, 0, 0], path, [[0, 6, 7, 90, 90, 90]]])
        
        # Need to check each step: move x, move y, move z, then rotate one axis
        # For simplicity and speed, check at each full pose
        for pose in full_poses:
            x, y, z, yaw, pitch, roll = pose
            R = rotation_matrix(yaw, pitch, roll)
            placed = R @ p + np.array([x, y, z])
            if not in_corridor(placed):
                return False
        return True

    def evaluate_path_fast(path):
        """Count 4x4x4 blocks where all points survive."""
        # Check all grid points
        surviving = np.zeros((81, 21, 21), dtype=bool)
        for i in range(81):
            for j in range(21):
                for k in range(21):
                    p = np.array([grid_x[i], grid_y[j], grid_z[k]])
                    surviving[i, j, k] = point_survives(p, path)
        
        # Count 4x4x4 blocks
        count = 0
        for i in range(81 - 3):
            for j in range(21 - 3):
                for k in range(21 - 3):
                    if np.all(surviving[i:i+4, j:j+4, k:k+4]):
                        count += 1
        return count

    # Build initial path
    path = build_path()
    
    # Quick evaluation
    best_count = evaluate_path_fast(path)
    best_path = path.copy()
    
    # Local search: perturb poses to improve
    t_search = time.time()
    search_budget = min(400, deadline - t_search - 30)
    
    if search_budget > 10:
        # Simple local search
        for iteration in range(2000):
            if time.time() > deadline - 30:
                break
            
            idx = np.random.randint(len(best_path))
            trial = best_path.copy()
            
            # Perturb position
            trial[idx, 0] += np.random.randn() * 0.02
            trial[idx, 1] += np.random.randn() * 0.02
            trial[idx, 2] += np.random.randn() * 0.02
            
            # Perturb angles
            trial[idx, 3] += np.random.randn() * 1.0
            trial[idx, 4] += np.random.randn() * 1.0
            trial[idx, 5] += np.random.randn() * 1.0
            
            # Keep within reasonable bounds
            trial[idx, 0] = np.clip(trial[idx, 0], -3, 1)
            trial[idx, 1] = np.clip(trial[idx, 1], 0, 6)
            trial[idx, 2] = np.clip(trial[idx, 2], 0, 7)
            trial[idx, 3:6] = np.clip(trial[idx, 3:6], 0, 90)
            
            count = evaluate_path_fast(trial)
            if count > best_count:
                best_count = count
                best_path = trial
    
    return best_path

# For testing
if __name__ == "__main__":
    start = time.time()
    result = moving_sofa_3d_path()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s")
    print(f"Shape: {result.shape}")
    print(f"First few poses: {result[:3]}")
    print(f"Last few poses: {result[-3:]}")
BET: {mechanism: "local search with bounded perturbation around seed path can find better corner positions and rotation centers that allow more 4x4x4 blocks to survive all intermediate poses", q: 0.65, delta: [0.001, 0.015]}
