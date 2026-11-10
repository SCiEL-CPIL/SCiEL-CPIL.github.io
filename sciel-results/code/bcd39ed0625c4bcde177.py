import numpy as np
import time
from scipy.optimize import minimize
import cma

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path for the moving sofa problem.
    Strategy:
    1. Directly optimize the (x, y) coordinates for each integer angle degree from 1 to 89.
       This results in a high-dimensional search space (178 dimensions).
    2. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for robust optimization.
    3. Vectorize the "inside corridor" check:
       - Precompute grid points.
       - For a given set of (x, y, a) waypoints, compute the inside mask for all grid points
         and all waypoints simultaneously using broadcasting.
       - The area is the count of grid points that are inside ALL waypoints.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Grid setup
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = grid_points.shape[0]
    
    # Precompute grid point coordinates
    gx = grid_points[:, 0]
    gy = grid_points[:, 1]
    
    # We will optimize waypoints for angles 1 to 89 degrees.
    # The evaluator adds (0,0,0) at start and (-2,0,90) at end.
    # The path is: (0,0,0) -> (x1,y1,1) -> ... -> (x89,y89,89) -> (-2,0,90)
    # We need to ensure that the sofa is inside the corridor at ALL these placements.
    
    angles_deg = np.arange(1.0, 90.0) # 1 to 89
    n_angles = len(angles_deg)
    angles_rad = np.deg2rad(angles_deg)
    
    # Precompute cos and sin for all angles
    cos_a = np.cos(angles_rad) # shape (n_angles,)
    sin_a = np.sin(angles_rad) # shape (n_angles,)
    
    def evaluate_path(params):
        """
        params: array of shape (2 * n_angles,)
                [x1, y1, x2, y2, ..., x89, y89]
        Returns: area (float)
        """
        if time.time() > deadline:
            return -1.0 # Signal timeout/invalid
            
        # Reshape params to (n_angles, 2)
        xy = params.reshape(n_angles, 2)
        x_path = xy[:, 0]
        y_path = xy[:, 1]
        
        # We need to check if grid points are inside the corridor for ALL angles.
        # Inside condition for angle a, center (xc, yc):
        # u = dx * cos(a) + dy * sin(a)
        # v = -dx * sin(a) + dy * cos(a)
        # Inside if (0<=u<=1 and -3<=v<=1) OR (-3<=u<=1 and 0<=v<=1)
        
        # dx = gx - x_path[i]
        # dy = gy - y_path[i]
        
        # To vectorize over angles AND points:
        # x_path shape: (n_angles, 1)
        # y_path shape: (n_angles, 1)
        # gx shape: (1, n_points)
        # gy shape: (1, n_points)
        
        # dx shape: (n_angles, n_points)
        # dy shape: (n_angles, n_points)
        
        # This creates a large array (89 * 135000) ~ 12 million floats.
        # 12M * 8 bytes = ~100MB. This is feasible.
        
        x_path_col = x_path[:, np.newaxis] # (n_angles, 1)
        y_path_col = y_path[:, np.newaxis] # (n_angles, 1)
        
        gx_row = gx[np.newaxis, :] # (1, n_points)
        gy_row = gy[np.newaxis, :] # (1, n_points)
        
        dx = gx_row - x_path_col # (n_angles, n_points)
        dy = gy_row - y_path_col # (n_angles, n_points)
        
        # u = dx * cos_a + dy * sin_a
        # cos_a shape: (n_angles, 1)
        # sin_a shape: (n_angles, 1)
        cos_a_col = cos_a[:, np.newaxis]
        sin_a_col = sin_a[:, np.newaxis]
        
        u = dx * cos_a_col + dy * sin_a_col
        v = -dx * sin_a_col + dy * cos_a_col
        
        # Mask 1: 0 <= u <= 1 and -3 <= v <= 1
        mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        # Mask 2: -3 <= u <= 1 and 0 <= v <= 1
        mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        mask = mask1 | mask2
        
        # We need points that are inside for ALL angles.
        # mask shape: (n_angles, n_points)
        # inside_all[i] is True if mask[i] is True.
        # We want points j such that mask[i, j] is True for all i.
        # This is equivalent to: np.all(mask, axis=0)
        
        inside_all = np.all(mask, axis=0)
        
        return np.sum(inside_all) * 0.0001

    def objective(params):
        area = evaluate_path(params)
        if area < 0:
            return 1e6
        return -area

    # Initial guess
    # A reasonable path is a curve from (0,0) to (-2,0).
    # Let's interpolate linearly in x and y?
    # x(t) = -2t, y(t) = 0 is a straight line.
    # But the sofa needs to turn.
    # Let's try a circular arc or just a simple curve.
    # For now, start with a straight line in x, y=0.
    # x goes from 0 to -2.
    # t = angles / 90
    t_init = angles_deg / 90.0
    x_init = -2.0 * t_init
    y_init = np.zeros(n_angles)
    
    # Actually, a straight line x=-2t, y=0 might not work because the sofa rotates.
    # Let's add a slight curve.
    # y = 0.5 * sin(pi * t) ?
    y_init = 0.5 * np.sin(np.pi * t_init)
    
    init_params = np.concatenate([x_init, y_init])
    
    # CMA-ES setup
    # Population size is usually 4 + 3 * ln(n)
    # n = 178
    # popsize = 4 + 3 * 5.18 = ~19.5 -> 20
    
    # We need to be careful with time.
    # Each evaluation takes ~0.1s (vectorized).
    # 20 evals per iteration.
    # 100 iterations = 2000 evals = 200s.
    # This fits in the budget.
    
    options = {
        'maxiter': 100,
        'verbose': -2,
        'seed': 42,
        'tolfun': 1e-8
    }
    
    # Use CMA-ES
    # Note: cma.fmin expects the function to return a scalar.
    # We want to maximize area, so minimize -area.
    
    best_params = init_params.copy()
    best_score = -objective(init_params)
    
    try:
        es = cma.CMAEvolutionStrategy(init_params, 0.5, options)
        
        while not es.stop() and time.time() < deadline - 30:
            solutions = es.ask()
            fits = [objective(s) for s in solutions]
            
            # Check for timeout in the middle of a generation
            if time.time() > deadline - 30:
                break
                
            es.tell(solutions, fits)
            
            # Update best
            best_fit = min(fits)
            if best_fit < best_score:
                best_score = best_fit
                best_params = solutions[fits.index(best_fit)].copy()
                
    except Exception as e:
        pass
        
    # If CMA-ES didn't run or failed, we might have a bad result.
    # Let's also try a few random restarts if time permits?
    # Or just return the best found.
    
    # Construct the output path
    # The evaluator expects an array of (x, y, a) for intermediate steps.
    # We have optimized for angles 1 to 89.
    # We should return these points.
    
    x_out = best_params[0::2]
    y_out = best_params[1::2]
    
    # Ensure finite
    if not np.all(np.isfinite(x_out)) or not np.all(np.isfinite(y_out)):
         # Fallback
         t_out = angles_deg / 90.0
         x_out = -2.0 * t_out
         y_out = np.zeros(n_angles)
         
    path_out = np.column_stack((x_out, y_out, angles_deg))
    
    # Double check validity
    if path_out.shape[0] < 1:
        return np.array([[-1.0, 0.0, 45.0]])
        
    return path_out

if __name__ == "__main__":
    res = moving_sofa_2d()
    print(res.shape)
    print(res[:5])
