# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move. In the reference frame of the sofa
    the corridor sweeps over the plane; the sofa is then the intersection of every
    corridor placement along the path, so by construction it always clears the corridor.
    The objective is to maximize the area of that intersection.

    The evaluator prepends the start state (0, 0, 0) and appends the end state
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps, exactly as in the
    AlphaEvolve "single turn" corridor notebook.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    # Set up the sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]

    # Start and End states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    # We will use a fixed number of intermediate points.
    # Let's choose 45 points, spaced by 2 degrees.
    # Angles: 2, 4, ..., 88
    N_path_points = 45
    angles = np.arange(2, 90, 2) # 2 to 88 inclusive
    
    # Initial guess: Linear interpolation of x, y
    # x goes from 0 to -2, y stays 0
    # t goes from 0 to 1
    t = np.linspace(0, 1, N_path_points)
    x_init = -2.0 * t
    y_init = np.zeros(N_path_points)
    
    best_path_xy = np.column_stack((x_init, y_init))
    
    def evaluate_path_xy(path_xy):
        """
        Evaluate the area for a given set of (x, y) coordinates.
        Assumes angles are fixed at `angles`.
        """
        # Construct full path with start and end
        # Start: (0,0,0)
        # End: (-2,0,90)
        # Intermediate: (path_xy[:,0], path_xy[:,1], angles)
        
        # We need to simulate the densification to know the exact placements.
        # But since we chose angles with step 2, and the evaluator uses m = ceil(da),
        # for step 2, m=2. So it adds 1 intermediate point.
        # The intermediate point is the linear interpolation of (x,y,a).
        
        # Let's just build the full list of (x,y,a) as the evaluator would.
        placements = []
        
        # Start
        prev_x, prev_y, prev_a = 0.0, 0.0, 0.0
        placements.append((prev_x, prev_y, prev_a))
        
        # Intermediate points
        for i in range(len(path_xy)):
            curr_x, curr_y = path_xy[i]
            curr_a = angles[i]
            
            da = abs(curr_a - prev_a)
            m = max(1, int(np.ceil(da)))
            
            for k in range(1, m + 1):
                t = k / m
                ix = prev_x + t * (curr_x - prev_x)
                iy = prev_y + t * (curr_y - prev_y)
                ia = prev_a + t * (curr_a - prev_a)
                placements.append((ix, iy, ia))
            
            prev_x, prev_y, prev_a = curr_x, curr_y, curr_a
        
        # End
        end_x, end_y, end_a = -2.0, 0.0, 90.0
        da = abs(end_a - prev_a)
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            ix = prev_x + t * (end_x - prev_x)
            iy = prev_y + t * (end_y - prev_y)
            ia = prev_a + t * (end_a - prev_a)
            placements.append((ix, iy, ia))
        
        # Now evaluate the intersection
        inside_all = np.ones(N_points, dtype=bool)
        
        for x, y, a_deg in placements:
            a_rad = np.radians(a_deg)
            dx = P[:, 0] - x
            dy = P[:, 1] - y
            
            cos_a = np.cos(-a_rad)
            sin_a = np.sin(-a_rad)
            
            u = cos_a * dx - sin_a * dy
            v = sin_a * dx + cos_a * dy
            
            # Check if (u, v) is in base corridor
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_base = in_rect1 | in_rect2
            
            inside_all &= in_base
            
            # Early exit if empty
            if not np.any(inside_all):
                return 0.0
        
        return np.sum(inside_all) * 0.0001

    # Initial evaluation
    best_score = evaluate_path_xy(best_path_xy)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Local Search
    # We will perform a simple hill-climbing with random restarts or just long runs.
    # Given the dimensionality (90 variables), a simple coordinate descent or random perturbations works.
    
    np.random.seed(42)
    
    current_path_xy = best_path_xy.copy()
    current_score = best_score
    
    # Adaptive step size
    step_size = 0.1
    
    while time.time() < deadline:
        # Pick a random index to perturb
        idx = np.random.randint(N_path_points)
        
        # Perturb x and y
        new_path_xy = current_path_xy.copy()
        new_path_xy[idx, 0] += np.random.normal(0, step_size)
        new_path_xy[idx, 1] += np.random.normal(0, step_size)
        
        # Optional: Keep x,y within reasonable bounds to avoid huge errors
        # The corridor is in [-3, 1] roughly.
        # x, y should probably be in [-4, 2]
        new_path_xy[idx, 0] = np.clip(new_path_xy[idx, 0], -5, 2)
        new_path_xy[idx, 1] = np.clip(new_path_xy[idx, 1], -5, 2)
        
        new_score = evaluate_path_xy(new_path_xy)
        
        if new_score > current_score:
            current_score = new_score
            current_path_xy = new_path_xy
            # Reduce step size slightly on success? Or keep it?
            # step_size *= 0.99
        else:
            # If we fail often, reduce step size?
            pass
            
        # Every 1000 iterations, check if we should restart or adjust
        # For now, just keep going.
        
        # Periodically update best
        if current_score > best_score:
            best_score = current_score
            best_path_xy = current_path_xy.copy()
            
    # Construct the final path array to return
    # The function must return an array of shape (n, 3)
    final_path = np.column_stack((best_path_xy[:, 0], best_path_xy[:, 1], angles))
    
    return final_path


# EVOLVE-BLOCK-END
