# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees)
    """
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill

    # Pre-compute grid points
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))  # (N, 2)
    N = P.shape[0]

    # Start and end states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Densify the path by adding linear interpolations between consecutive rows.
        """
        if len(path) == 0:
            return np.array([start_state, end_state])
        
        full_path = [start_state] + list(path) + [end_state]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                densified.append([x, y, a])
        
        return np.array(densified)

    def evaluate_path(path):
        """
        Evaluate the area of the sofa for a given path.
        Vectorized over all grid points for each placement.
        """
        densified = densify_path(path)
        
        in_sofa = np.ones(N, dtype=bool)
        
        for x, y, a in densified:
            a_rad = np.radians(a)
            cos_a = np.cos(-a_rad)
            sin_a = np.sin(-a_rad)
            
            dx = P[:, 0] - x
            dy = P[:, 1] - y
            
            u = cos_a * dx - sin_a * dy
            v = sin_a * dx + cos_a * dy
            
            in_r1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
            in_r2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
            
            mask = in_r1 | in_r2
            in_sofa &= mask
            
            if not in_sofa.any():
                break
        
        return in_sofa.sum() * 0.0001

    # Multiple initial paths to try
    initial_paths = [
        np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64),
        np.array([
            [-0.05, 0.05, 10.0],
            [-0.15, 0.15, 25.0],
            [-0.25, 0.25, 40.0],
            [-0.35, 0.25, 55.0],
            [-0.45, 0.2, 70.0],
            [-0.55, 0.2, 85.0],
        ], dtype=np.float64),
        np.array([
            [-0.15, 0.15, 12.0],
            [-0.25, 0.2, 22.0],
            [-0.3, 0.3, 30.0],
            [-0.35, 0.3, 45.0],
            [-0.4, 0.3, 55.0],
            [-0.5, 0.2, 65.0],
            [-0.6, 0.2, 78.0],
            [-0.65, 0.25, 85.0],
        ], dtype=np.float64),
    ]

    best_path = None
    best_score = 0.0

    for init_path in initial_paths:
        if (time.time() - start_time) > deadline - 60:
            break

        path = init_path.copy()
        score = evaluate_path(path)
        
        if score > best_score:
            best_score = score
            best_path = path.copy()

        # Local search with adaptive step sizes
        np.random.seed(int(time.time() * 1000) % 10000)
        
        step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01]
        
        for step_size in step_sizes:
            if (time.time() - start_time) > deadline - 30:
                break
            
            improved = True
            while improved and (time.time() - start_time) < deadline - 15:
                improved = False
                for i in range(len(path)):
                    if (time.time() - start_time) > deadline - 15:
                        break
                    
                    for _ in range(30):  # More perturbations per point
                        if (time.time() - start_time) > deadline - 15:
                            break
                        
                        new_path = path.copy()
                        new_path[i, 0] += np.random.uniform(-step_size, step_size)
                        new_path[i, 1] += np.random.uniform(-step_size, step_size)
                        new_path[i, 2] += np.random.uniform(-step_size * 1.5, step_size * 1.5)
                        
                        # Ensure angles are in reasonable range
                        new_path[i, 2] = np.clip(new_path[i, 2], 2, 88)
                        
                        new_score = evaluate_path(new_path)
                        if new_score > score:
                            score = new_score
                            path = new_path
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_path = path.copy()

    # Final validation
    if best_path is None or not np.all(np.isfinite(best_path)):
        best_path = initial_paths[0]

    return best_path

# EVOLVE-BLOCK-END
