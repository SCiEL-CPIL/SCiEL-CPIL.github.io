import numpy as np
from scipy.optimize import differential_evolution

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the L-shaped corridor to maximize the area of the intersection
    of all corridor placements using Differential Evolution (Global Optimization).
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

    def calculate_area(path, grid_size=150):
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
        
        # Use a moderately fine grid for optimization
        # 150x150 is a good balance. The actual grid is 450x300.
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
        
        # Approx area
        # Grid spacing is 4/(grid_size-1)
        area = np.sum(mask) * (4.0 / (grid_size - 1)) ** 2
        return area

    def objective(params):
        x1, y1, a1, x2, y2, a2 = params
        # Ensure angles are monotonic? 
        # a1, a2 should be between 0 and 90
        # If they are not, the path might be weird, but the evaluator handles it.
        # However, for this specific problem, non-monotonic angles are likely suboptimal.
        # We can penalize or just let the optimizer figure it out.
        # Let's keep bounds strict to 0..90.
        
        path = np.array([
            [0.0, 0.0, 0.0],
            [x1, y1, a1],
            [x2, y2, a2],
            [-2.0, 0.0, 90.0]
        ])
        
        # Calculate area
        area = calculate_area(path, grid_size=150)
        
        # We want to maximize area, so minimize negative area
        return -area

    # Bounds for Differential Evolution
    # x1, y1, a1, x2, y2, a2
    bounds = [
        (-3.0, 3.0), # x1
        (-3.0, 3.0), # y1
        (0.0, 90.0), # a1
        (-3.0, 3.0), # x2
        (-3.0, 3.0), # y2
        (0.0, 90.0)  # a2
    ]

    # Run Differential Evolution
    # maxiter and popsize control the search effort.
    # 600s limit is generous.
    # DE is stochastic, so we might need a few iterations.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=True # Use Nelder-Mead to polish the result
        )
        
        if result.success:
            x1, y1, a1, x2, y2, a2 = result.x
            path = np.array([
                [0.0, 0.0, 0.0],
                [x1, y1, a1],
                [x2, y2, a2],
                [-2.0, 0.0, 90.0]
            ])
        else:
            # Fallback
            path = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 45.0],
                [0.0, 0.0, 90.0],
                [-2.0, 0.0, 90.0]
            ])
    except Exception:
        # Fallback
        path = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 45.0],
            [0.0, 0.0, 90.0],
            [-2.0, 0.0, 90.0]
        ])

    # Extract intermediate points
    if len(path) > 2:
        return path[1:-1]
    else:
        return np.array([[0.0, 0.0, 45.0]])

# EVOLVE-BLOCK-END
