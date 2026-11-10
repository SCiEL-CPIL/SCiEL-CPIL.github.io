# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

# Precompute the sofa grid points
# i = 0..449, j = 0..299
# x in [-3, 1.5), y in [-1.5, 1.5)
x_grid = np.arange(-3.0, 1.5, 0.01)
y_grid = np.arange(-1.5, 1.5, 0.01)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
# Reshape to (N, 2)
grid_points = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(grid_points)

def densify_path(path):
    """
    Densify the path according to the evaluator's rules.
    path: (M, 3) array of (x, y, a).
    Returns: (K, 3) array of densified placements.
    """
    if len(path) == 0:
        return np.empty((0, 3))
    
    placements = [path[0]]
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        # Generate m+1 points from p1 to p2 (inclusive)
        # t = np.linspace(0, 1, m + 1)
        # pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
        # Avoid duplicating p1 which is already added
        # Actually, just add the new points from index 1 to m
        if m > 1:
            t = np.linspace(0, 1, m + 1)[1:]  # 1/m, 2/m, ..., 1
            pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
            placements.extend(pts)
        else:
            # m=1, just add p2
            placements.append(p2)
            
    return np.array(placements)

def compute_area(path_params):
    """
    path_params: flat array of length 3*N
    Returns: area (float)
    """
    N = len(path_params) // 3
    path = path_params.reshape(N, 3)
    
    # Construct full path with start and end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full_path = np.vstack([start, path, end])
    
    # Densify
    placements = densify_path(full_path)
    
    if len(placements) == 0:
        return 0.0

    # Check if grid points are inside ALL placements
    # W: (N_POINTS, 2)
    # P: (K, 3)
    
    # Initialize mask as all True
    inside_all = np.ones(N_POINTS, dtype=bool)
    
    for p in placements:
        x, y, a_deg = p
        a_rad = np.deg2rad(a_deg)
        
        # Rotate grid points by -a and translate by -p
        # w_local = R(-a) * (w - p)
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        # Note: Standard rotation R(theta) = [[cos, -sin], [sin, cos]].
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = grid_points[:, 0] - x
        dy = grid_points[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Base corridor:
        # Rect 1: 0 <= u <= 1, -3 <= v <= 1
        # Rect 2: -3 <= u <= 1, 0 <= v <= 1
        
        in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        in_corridor = in_r1 | in_r2
        
        inside_all &= in_corridor
        
        # Early termination if no points are inside
        if not np.any(inside_all):
            return 0.0

    return np.sum(inside_all) * 0.0001

def objective(params):
    """
    Minimize negative area.
    """
    try:
        area = compute_area(params)
        return -area
    except:
        return 1e9 # Large penalty for errors

def get_initial_guesses(num_guesses=5):
    guesses = []
    # Guess 1: Linear interpolation of the 5 intermediate points
    # Start (0,0,0), End (-2,0,90)
    # Interpolate x, y, a
    ts = np.linspace(0, 1, 7)[1:-1] # 5 points
    x = ts * -2.0
    y = np.zeros(5)
    a = ts * 90.0
    guesses.append(np.column_stack((x, y, a)).ravel())
    
    # Guess 2: Arc-like
    # Try to curve the path
    for curvature in [0.5, -0.5, 1.0]:
        x = ts * -2.0 + curvature * np.sin(np.pi * ts)
        y = curvature * np.cos(np.pi * ts) # Just a perturbation
        # Ensure y is small
        y = y * 0.1 
        a = ts * 90.0
        guesses.append(np.column_stack((x, y, a)).ravel())
        
    return guesses

def moving_sofa_2d() -> np.ndarray:
    """
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees)
    """
    start_time = time.time()
    deadline = start_time + 580.0 # 20s safety margin
    
    best_params = None
    best_score = -1.0
    
    # Define bounds for the 5 intermediate points
    # x: [-5, 5], y: [-5, 5], a: [0, 90]
    bounds = []
    for _ in range(5):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 90.0)) # a
        
    # Get initial guesses
    guesses = get_initial_guesses()
    
    # Shuffle guesses to ensure diversity
    np.random.shuffle(guesses)
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        # Run optimization
        # Nelder-Mead is robust
        try:
            res = minimize(
                objective, 
                guess, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={
                    'maxiter': 500, 
                    'xatol': 1e-4, 
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception as e:
            pass # Ignore errors in optimization
            
        # Check time
        if time.time() > deadline - 60.0:
            break
            
    if best_params is None:
        # Fallback to baseline
        return np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
        
    # Reshape to (5, 3)
    best_path = best_params.reshape(5, 3)
    
    return best_path

# EVOLVE-BLOCK-END
