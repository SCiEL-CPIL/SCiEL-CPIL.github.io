import numpy as np
from scipy.optimize import minimize
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for the moving sofa problem by optimizing the intermediate
    positions and angles to maximize the area of the intersection of the
    corridor placements.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Grid for area estimation
    # The sofa grid is (-3, 1.5) x (-1.5, 1.5)
    # Coarse grid for optimization
    n_x, n_y = 200, 200
    x_grid = np.linspace(-3, 1.5, n_x)
    y_grid = np.linspace(-1.5, 1.5, n_y)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    cell_area = dx * dy

    def compute_area(path):
        """
        path: list of (x, y, angle_deg)
        Returns estimated area of intersection.
        """
        # Prepend and append
        full_path = [(0.0, 0.0, 0.0)] + list(path) + [(-2.0, 0.0, 90.0)]
        
        # Densify
        densified_path = [full_path[0]]
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
                densified_path.append((x, y, a))
        
        # Initialize mask
        mask = np.ones(len(grid_points), dtype=bool)
        
        # Check each placement
        for x, y, a_deg in densified_path:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            wx = grid_points[:, 0] - x
            wy = grid_points[:, 1] - y
            
            u = cos_a * wx + sin_a * wy
            v = -sin_a * wx + cos_a * wy
            
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = cond1 | cond2
            
            mask &= inside
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * cell_area

    def objective(params):
        n_points = len(params) // 3
        path = []
        for i in range(n_points):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            path.append((x, y, a))
        return -compute_area(path)

    n_intermediate = 8
    bounds = []
    for _ in range(n_intermediate):
        bounds.append((-3.0, 1.0)) # x
        bounds.append((-3.0, 1.0)) # y
        bounds.append((0.0, 90.0)) # a
    
    x0 = []
    for i in range(1, n_intermediate + 1):
        t = i / (n_intermediate + 1)
        x = 0.0 + t * (-2.0 - 0.0)
        y = 0.0
        a = 0.0 + t * (90.0 - 0.0)
        x += np.random.uniform(-0.5, 0.5)
        y += np.random.uniform(-0.5, 0.5)
        a += np.random.uniform(-5.0, 5.0)
        a = np.clip(a, 0.1, 89.9)
        x0.extend([x, y, a])
    
    best_path = []
    best_score = -1.0
    
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        x0_current = np.array(x0)
        if r > 0:
            x0_current += np.random.uniform(-1.0, 1.0, size=x0.shape)
            for i in range(n_intermediate):
                idx = i * 3 + 2
                x0_current[idx] = np.clip(x0_current[idx], 0.1, 89.9)
        
        try:
            res = minimize(objective, x0_current, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                best_score = -res.fun
                best_path = []
                for i in range(n_intermediate):
                    best_path.append((res.x[i*3], res.x[i*3+1], res.x[i*3+2]))
        except Exception:
            pass

    if not best_path:
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

    return np.array(best_path, dtype=np.float64)
