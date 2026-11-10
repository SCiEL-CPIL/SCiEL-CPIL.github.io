import numpy as np
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the L-shaped corridor to maximize the area of the intersection
    of all corridor placements.
    """
    
    # Base corridor definition
    # u, v are local coordinates
    # Base is union of [0,1]x[-3,1] and [-3,1]x[0,1]
    
    def is_inside_base(u, v):
        # Vectorized check
        # Part 1: 0 <= u <= 1, -3 <= v <= 1
        cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        # Part 2: -3 <= u <= 1, 0 <= v <= 1
        cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return cond1 | cond2

    def calculate_area(path):
        # Densify path
        # Evaluator logic: m = max(1, ceil(|a2 - a1|))
        # We replicate this to get the actual placements
        
        placements = [path[0]]
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m+1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append([x, y, a])
        
        # Coarse grid for speed
        # Grid size 100x100 for optimization, 450x300 for final check?
        # Let's use 100x100 for optimization speed.
        grid_size = 100
        u = np.linspace(-3, 1, grid_size)
        v = np.linspace(-3, 1, grid_size)
        U, V = np.meshgrid(u, v)
        points = np.column_stack((U.ravel(), V.ravel()))
        
        # Check intersection
        # Start with all True
        mask = np.ones(len(points), dtype=bool)
        
        for p in placements:
            x, y, a_deg = p
            a_rad = np.deg2rad(a_deg)
            # R(-a)
            cos_a = np.cos(-a_rad)
            sin_a = np.sin(-a_rad)
            
            # w - (x,y)
            dx = points[:, 0] - x
            dy = points[:, 1] - y
            
            # Rotate
            u_loc = cos_a * dx - sin_a * dy
            v_loc = sin_a * dx + cos_a * dy
            
            inside = is_inside_base(u_loc, v_loc)
            mask &= inside
            
            # Early exit if empty
            if not np.any(mask):
                return 0.0
        
        area = np.sum(mask) * (4.0 / (grid_size - 1)) ** 2 # Approx area
        return area

    # Initial guess: Rotate then Slide
    # (0,0,0) -> (0,0,90) -> (-2,0,90)
    # Intermediate points
    # Let's optimize 2 intermediate points
    # x1, y1, a1, x2, y2, a2
    
    def objective(params):
        x1, y1, a1, x2, y2, a2 = params
        # Ensure angles are monotonic? Not strictly required, but helps.
        # a1, a2 should be between 0 and 90
        path = np.array([
            [0.0, 0.0, 0.0],
            [x1, y1, a1],
            [x2, y2, a2],
            [-2.0, 0.0, 90.0]
        ])
        # Basic bounds
        if not (0 <= a1 <= 90 and 0 <= a2 <= 90):
            return -1
        if a2 < a1:
            return -1
        return -calculate_area(path) # Minimize negative area

    # Initial guess
    x0 = [0.0, 0.0, 45.0, -2.0, 0.0, 90.0] # Rotate at 0, then slide
    
    # Bounds
    bounds = [
        (-3.0, 3.0), # x1
        (-3.0, 3.0), # y1
        (0.0, 90.0), # a1
        (-3.0, 3.0), # x2
        (-3.0, 3.0), # y2
        (0.0, 90.0)  # a2
    ]

    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                      options={'maxiter': 200, 'xatol': 0.1, 'fatol': 0.01})
    
    if result.success:
        x1, y1, a1, x2, y2, a2 = result.x
        path = np.array([
            [0.0, 0.0, 0.0],
            [x1, y1, a1],
            [x2, y2, a2],
            [-2.0, 0.0, 90.0]
        ])
    else:
        # Fallback to Rotate-Then-Slide
        path = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 45.0],
            [0.0, 0.0, 90.0],
            [-2.0, 0.0, 90.0]
        ])
        
    # Ensure endpoints are included in the returned path?
    # The evaluator adds (0,0,0) and (-2,0,90).
    # So we should return the intermediate points only?
    # "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
    # So we should return the path WITHOUT the endpoints.
    
    # My `path` above includes endpoints.
    # Extract intermediate points
    if len(path) > 2:
        return path[1:-1]
    else:
        # If only endpoints, return empty? But n>=1.
        # Add a midpoint
        return np.array([[0.0, 0.0, 45.0]])

# EVOLVE-BLOCK-END
