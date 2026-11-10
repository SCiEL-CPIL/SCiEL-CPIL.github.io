# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Direct waypoint optimization.
    We represent the path by 89 waypoints (angles 1 to 89 degrees).
    We use a vectorized evaluation function to check all waypoints simultaneously.
    We perform a stochastic local search on the (x, y) coordinates of these waypoints.
    """
    
    # 1. Setup Grid for Evaluation
    # Sofa points: x in [-3, 1.5), y in [-1.5, 1.5), step 0.01
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])
    
    # Waypoints: angles 1 to 89
    angles = np.arange(1, 90)
    n_wp = len(angles)
    
    # Initial path: Linear interpolation in x, y=0
    # x goes from 0 to -2, so x(a) = -2 * a / 90
    x_init = -2.0 * angles / 90.0
    y_init = np.zeros(n_wp)
    
    def evaluate_path_vectorized(x_wp, y_wp):
        """
        Evaluates the area of the sofa for a given path defined by waypoints.
        x_wp, y_wp: arrays of shape (n_wp,)
        Returns: float area
        """
        # We need to check the corridor constraint for all interpolated steps.
        # The evaluator adds linear interpolations between consecutive rows.
        # Since we have waypoints at every integer degree, the interpolation is just between angle a and a+1.
        # The evaluator adds points at t = j/steps. steps = max(1, ceil(|a2-a1|)).
        # Here a2-a1 is always 1 (except boundaries). So steps = 1.
        # This means the evaluator only checks the endpoints of our segments?
        # "Between consecutive rows it adds linear interpolations ... in m = max(1, ceil(|a2 - a1|)) equal steps"
        # If we provide points at a=1, a=2, ..., a=89.
        # Segment 1: (0,0,0) to (x1, y1, 1). da=1. steps=1. Adds p2 (the first waypoint).
        # Segment 2: (x1, y1, 1) to (x2, y2, 2). da=1. steps=1. Adds p2 (the second waypoint).
        # ...
        # Segment 89: (x88, y88, 88) to (x89, y89, 89). da=1. steps=1. Adds p2.
        # Segment 90: (x89, y89, 89) to (-2, 0, 90). da=1. steps=1. Adds end_state.
        
        # So effectively, the evaluator checks:
        # 1. start_state (0,0,0)
        # 2. All our waypoints (x_wp, y_wp, angles)
        # 3. end_state (-2, 0, 90)
        
        # Wait, does it check the intermediate points if steps > 1?
        # Yes, but since we define waypoints at every integer degree, steps is always 1.
        # So we only need to check the states at integer angles.
        
        # Let's construct the full list of states to check.
        # State 0: start
        # States 1..89: waypoints
        # State 90: end
        
        # To be safe and efficient, we can vectorize the check for all these states.
        # However, the mask is a boolean array of size N_points.
        # We need: mask_final = mask_start & mask_wp1 & ... & mask_wp89 & mask_end
        
        # Pre-compute rotation matrices? No, angles are different for each state.
        # We have 91 states to check.
        # N_points is ~135k.
        # 91 * 135k operations is ~12M operations. This is fast in NumPy.
        
        mask = np.ones(N_points, dtype=bool)
        
        # Helper to apply one state
        def apply_state(x, y, a_deg):
            nonlocal mask
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = wx - x
            dy = wy - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            in_corridor = in_r1 | in_r2
            
            mask &= in_corridor
            
            # Early exit if empty
            if not np.any(mask):
                return False
            return True
        
        # Check Start
        if not apply_state(0.0, 0.0, 0.0):
            return 0.0
            
        # Check Waypoints
        for i in range(n_wp):
            if not apply_state(x_wp[i], y_wp[i], angles[i]):
                return 0.0
                
        # Check End
        if not apply_state(-2.0, 0.0, 90.0):
            return 0.0
            
        return float(np.sum(mask)) * 0.0001

    # 2. Optimization Loop
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Current best
    best_x = x_init.copy()
    best_y = y_init.copy()
    best_score = evaluate_path_vectorized(best_x, best_y)
    
    # Stochastic Local Search
    # We perturb the waypoints.
    # We want to keep the path somewhat smooth, but allow flexibility.
    # We can use a "heat" parameter that decays over time.
    
    current_x = x_init.copy()
    current_y = y_init.copy()
    current_score = best_score
    
    step_size = 0.5
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Perturb a subset of waypoints or all of them?
        # Perturbing all might be too chaotic. Let's perturb all with small noise.
        # Or, pick a random window of waypoints to perturb to allow local shape changes.
        
        # Let's try perturbing all waypoints with Gaussian noise.
        # We also need to ensure the path doesn't jump too wildly.
        
        new_x = current_x + np.random.normal(0, step_size, n_wp)
        new_y = current_y + np.random.normal(0, step_size, n_wp)
        
        # Optional: Smooth the path slightly to avoid high-frequency noise that might cause
        # interpolation issues? No, the evaluator only checks integer angles if steps=1.
        # But wait, if we return the path, the evaluator interpolates.
        # If we return points at 1, 2, ..., 89, the interpolation is linear between them.
        # The evaluator checks the endpoints of these linear segments.
        # So the "shape" between integer angles is a straight line in (x,y,a) space.
        # This is fine.
        
        new_score = evaluate_path_vectorized(new_x, new_y)
        
        if new_score > current_score:
            current_x = new_x
            current_y = new_y
            current_score = new_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_x = current_x.copy()
                best_y = current_y.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reduce step size
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 0.01:
                    # Try a larger jump to escape local optimum?
                    # Or just break if we've been stuck for a long time
                    if step_size < 0.001:
                        break

    # Final check: Ensure the returned path is valid and finite
    # The function requires a float array of shape (n, 3).
    # We return the waypoints.
    # Shape: (89, 3)
    
    result = np.column_stack((best_x, best_y, angles))
    
    # Ensure all finite
    if not np.all(np.isfinite(result)):
        # Fallback
        return np.array([[0.0, 0.0, 0.0]])
        
    return result

# EVOLVE-BLOCK-END
