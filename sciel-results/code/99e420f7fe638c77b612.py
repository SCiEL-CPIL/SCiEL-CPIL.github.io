# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for the moving sofa problem that maximizes the sofa area.
    
    Strategy: Coarse-to-fine grid search over the configuration space.
    We discretize the angle from 0 to 90 degrees. For each angle, we find the
    (x, y) position that maximizes the number of sofa points remaining in the
    corridor. We use a coarse grid first to find the general region, then refine.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Grid setup for sofa points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = grid_points.shape[0]
    
    def evaluate_path(path):
        """
        Evaluate the sofa area for a given path.
        path: array of shape (n, 3) with (x, y, angle_degrees)
        Returns the raw value (number of sofa points * 0.0001)
        """
        # Build the full path with endpoints and interpolation
        full_path = [np.array([0.0, 0.0, 0.0])]
        
        for i in range(len(path)):
            full_path.append(path[i])
        
        full_path.append(np.array([-2.0, 0.0, 90.0]))
        
        # Convert to array
        full_path = np.array(full_path)
        
        # Interpolate between consecutive points
        interp_paths = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                ts = np.linspace(0, 1, m + 1)[1:]  # Skip t=0 (already added)
                for t in ts:
                    interp = p1 + t * (p2 - p1)
                    interp_paths.append(interp)
        
        interp_paths = np.array(interp_paths)
        
        # For each grid point, check if it's in ALL corridor placements
        # Start with all True
        inside_all = np.ones(n_points, dtype=bool)
        
        for placement in interp_paths:
            x, y, a_deg = placement
            a_rad = np.radians(a_deg)
            
            # R(-a) rotation matrix
            c = np.cos(-a_rad)
            s = np.sin(-a_rad)
            
            # w - (x, y)
            dx = grid_points[:, 0] - x
            dy = grid_points[:, 1] - y
            
            # Rotate
            u = c * dx - s * dy
            v = s * dx + c * dy
            
            # Check if in base
            in_v = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_h = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_base = in_v | in_h
            
            inside_all &= in_base
            
            # Early exit if no points left
            if not np.any(inside_all):
                return 0.0
        
        n_inside = np.sum(inside_all)
        return n_inside * 0.0001

    # --- Grid Search Strategy ---
    
    # We will search for the best (x, y) for a sequence of angles.
    # The evaluator interpolates between our points. 
    # If we provide points at every integer degree, the interpolation is exact (m=1).
    # So we can just search for the best (x, y) for each angle 1..89.
    
    # However, searching 89 angles independently is not quite right because 
    # the "sofa" is the intersection of ALL placements. 
    # If we pick the optimal (x, y) for angle 10 independently of angle 11, 
    # the intersection might be smaller than if we picked a slightly sub-optimal 
    # (x, y) for angle 10 that was better for angle 11.
    
    # A better approach: 
    # 1. Fix the angles to be 1, 2, ..., 89.
    # 2. Use a dynamic programming or iterative refinement approach.
    # 3. Or, simply use a random search over the entire path but with a better initial guess 
    #    derived from a coarse grid search on a subset of angles.
    
    # Let's try a "Sweep" approach:
    # We know the path must go from (0,0,0) to (-2,0,90).
    # Let's discretize the angle into N steps.
    # For each step, we want to find the (x,y) that maximizes the intersection.
    
    # Since the objective is non-convex and coupled, we will use a multi-start random search
    # but initialized with a coarse grid search result for a few key angles.
    
    # Key angles: 0, 45, 90.
    # Let's find a good (x, y) for 45 degrees first.
    
    # Coarse grid for x, y
    x_range = np.arange(-2.0, 0.5, 0.1)
    y_range = np.arange(-0.5, 1.0, 0.1)
    
    best_path = np.array([])
    
    # Helper to evaluate a specific path
    def get_score(path):
        return evaluate_path(path)
    
    # Initial best score
    best_score = 0.0
    
    # Phase 1: Coarse search for a single intermediate point (e.g., at 45 degrees)
    # Path: (0,0,0) -> (x, y, 45) -> (-2,0,90)
    
    for x in x_range:
        for y in y_range:
            if time.time() > deadline - 100: break
            path = np.array([[x, y, 45.0]])
            score = get_score(path)
            if score > best_score:
                best_score = score
                best_path = path
    
    if len(best_path) == 0:
        # Fallback to seed-like path
        best_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ])
        best_score = get_score(best_path)
    
    # Phase 2: Refine with more points and local search
    # Let's expand the path to have points at every 15 degrees.
    # Angles: 15, 30, 45, 60, 75
    # We will use the coarse result to initialize the 45 degree point, and neighbors.
    
    # Initialize a path with 5 points
    # (0,0,0) is start, (-2,0,90) is end.
    # We control points at 15, 30, 45, 60, 75.
    
    # Use the best (x,y) from Phase 1 for 45.
    x45, y45 = best_path[0, 0], best_path[0, 1]
    
    # For 15 and 30, guess linearly or use similar x,y
    # For 60 and 75, guess linearly towards (-2,0)
    
    init_x = np.array([x45 - 0.2, x45 - 0.1, x45, x45 - 0.1, x45 - 0.2])
    init_y = np.array([y45 - 0.1, y45 - 0.05, y45, y45 - 0.05, y45 - 0.1])
    angles = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
    
    best_path = np.column_stack((init_x, init_y, angles))
    best_score = get_score(best_path)
    
    # Local search with random perturbations
    np.random.seed(42)
    
    n_iter = 0
    while time.time() < deadline:
        # Perturb the path
        perturbed = best_path.copy()
        
        # Randomly decide how much to perturb
        # Sometimes small steps, sometimes larger
        scale = np.random.choice([0.05, 0.1, 0.2, 0.5])
        
        for i in range(len(perturbed)):
            perturbed[i, 0] += np.random.normal(0, scale)  # x
            perturbed[i, 1] += np.random.normal(0, scale)  # y
            # Angles are fixed in this strategy to ensure we cover the range densely
            # If we allow angles to vary, we need to be careful with interpolation.
            # Keeping angles fixed at 15, 30, 45, 60, 75 ensures m=1 interpolation 
            # if we were to add more points, but here we just have these 5.
            # Actually, the evaluator interpolates between our points.
            # If we keep angles fixed, the path is well-defined.
            
        # Ensure angles are sorted and within range if we allowed them to change.
        # Here we keep them fixed.
        
        score = get_score(perturbed)
        
        if score > best_score:
            best_score = score
            best_path = perturbed.copy()
        
        n_iter += 1
        
        # If we have plenty of time, try adding more points?
        # Adding more points allows for a smoother turn, potentially increasing area.
        # Let's try to add a point if we are stuck for a while.
        # For now, stick to 5 points to keep evaluation fast.
        
    return best_path

# EVOLVE-BLOCK-END
