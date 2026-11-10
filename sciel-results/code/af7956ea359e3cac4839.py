import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path for the moving sofa problem.
    Strategy:
    1. Parameterize the path (x, y) as smooth functions of angle a (0 to 90).
       We use a polynomial representation: x(a) = sum c_i * (a/90)^i, y(a) = sum d_i * (a/90)^i.
       This ensures smoothness and reduces the search space dimensionality.
    2. Evaluate the intersection area by checking a dense set of angles.
    3. Use scipy.optimize to maximize the area.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Grid setup
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = grid_points.shape[0]
    
    # Precompute grid point coordinates as columns for broadcasting
    gx = grid_points[:, 0]
    gy = grid_points[:, 1]

    def calculate_area(path_params, n_angles=180):
        """
        path_params: array of shape (2 * (deg+1),) 
                     [c0, c1, ..., c_deg, d0, d1, ..., d_deg]
        n_angles: number of angles to sample for evaluation
        """
        deg = 3 # Degree of polynomial
        n_coeffs = deg + 1
        cx = path_params[:n_coeffs]
        cy = path_params[n_coeffs:]
        
        # Angles from 0 to 90 degrees
        angles_deg = np.linspace(0.0, 90.0, n_angles)
        angles_rad = np.deg2rad(angles_deg)
        
        # Normalize angle t in [0, 1]
        t = angles_deg / 90.0
        
        # Evaluate polynomials
        # x(t) = c0 + c1*t + c2*t^2 + c3*t^3
        # y(t) = d0 + d1*t + d2*t^2 + d3*t^3
        powers = np.vstack([t**i for i in range(n_coeffs)]).T # shape (n_angles, n_coeffs)
        xs_path = powers @ cx
        ys_path = powers @ cy
        
        # Check intersection
        # We need to check if all grid points are inside the corridor at ALL sampled angles.
        # Instead of iterating over angles in Python, we can vectorize the check.
        # However, the "inside" condition is non-linear (rotation).
        # Let's loop over angles but vectorize over grid points.
        # n_angles=180 is fast enough.
        
        inside_all = np.ones(n_points, dtype=bool)
        
        for i in range(n_angles):
            a_rad = angles_rad[i]
            x_c = xs_path[i]
            y_c = ys_path[i]
            
            dx = gx - x_c
            dy = gy - y_c
            
            c = np.cos(a_rad)
            s = np.sin(a_rad)
            
            u = dx * c + dy * s
            v = -dx * s + dy * c
            
            mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            mask = mask1 | mask2
            
            inside_all &= mask
            
            # Early exit if empty
            if not np.any(inside_all):
                return 0.0
                
        return np.sum(inside_all) * 0.0001

    def objective(params):
        # Maximize area -> Minimize negative area
        area = calculate_area(params)
        return -area

    # Initial guess
    # Baseline path from previous iterations suggests a curve that moves left and slightly up/down.
    # Let's initialize with a simple linear or quadratic path.
    # x goes from 0 to -2, y goes from 0 to 0 roughly, but curves.
    # Let's try to fit the baseline points to a polynomial.
    # Baseline:
    # (0,0,0) -> (-0.1, 0.1, 15) -> ... -> (-2, 0, 90)
    
    # Let's use a fixed set of angles to fit the baseline points to get initial coeffs
    base_angles = np.array([0, 15, 20, 34, 40, 50, 55, 75, 86, 90])
    base_x = np.array([0, -0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7, -2.0])
    base_y = np.array([0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3, 0.0])
    
    t_fit = base_angles / 90.0
    V = np.vstack([t_fit**i for i in range(4)]).T
    
    # Least squares fit for x and y
    try:
        cx_init, _, _, _ = np.linalg.lstsq(V, base_x, rcond=None)
        cy_init, _, _, _ = np.linalg.lstsq(V, base_y, rcond=None)
    except:
        cx_init = np.array([0.0, -2.0, 0.0, 0.0])
        cy_init = np.array([0.0, 0.0, 0.5, -0.5])
        
    init_params = np.concatenate([cx_init, cy_init])
    
    # Evaluate initial
    current_score = -objective(init_params)
    best_params = init_params.copy()
    
    # Optimization
    # Use Nelder-Mead or L-BFGS-B. 
    # The objective is non-smooth due to grid discretization, but the underlying geometry is smooth.
    # Nelder-Mead is robust to non-smoothness.
    
    options = {
        'maxiter': 1000,
        'xatol': 1e-4,
        'fatol': 1e-6
    }
    
    # We will run multiple optimizations with different starting points or methods if time permits.
    # First, try Nelder-Mead on the initial guess.
    
    result = minimize(objective, init_params, method='Nelder-Mead', options=options)
    
    if result.success or result.fun < objective(init_params):
        new_score = -result.fun
        if new_score > current_score:
            current_score = new_score
            best_params = result.x.copy()
            
    # Try a few random restarts if time allows
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
            
        # Perturb initial params
        noise = np.random.normal(0, 0.1, size=best_params.shape)
        perturbed_params = best_params + noise
        
        # Also try completely different random starts occasionally
        if i == n_restarts - 1:
             perturbed_params = np.random.normal(0, 0.5, size=8)
             # Force endpoints roughly? No, the polynomial doesn't strictly enforce endpoints.
             # We should add constraints or penalty if endpoints are wrong.
             # However, the evaluator adds (0,0,0) and (-2,0,90).
             # Our polynomial defines the *intermediate* path? 
             # NO. The task says: "Row (x, y, a) places... The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
             # This means the *output* of moving_sofa_2d is the *intermediate* rows.
             # The evaluator constructs the full path: [(0,0,0)] + [my_rows] + [(-2,0,90)].
             # My `calculate_area` function assumed the path includes the endpoints.
             # I MUST adjust the strategy to only optimize the INTERMEDIATE points, or ensure the polynomial matches the endpoints.
             
             # Let's re-evaluate the parameterization.
             # If I use a polynomial for x(a) and y(a), it must satisfy:
             # x(0) = 0, y(0) = 0
             # x(90) = -2, y(90) = 0
             
             # So, let's define the path as:
             # x(a) = 0 + (a/90) * (-2 - 0) + (a/90)*(1 - a/90) * P_x(a)
             # y(a) = 0 + (a/90) * (0 - 0) + (a/90)*(1 - a/90) * P_y(a)
             # This guarantees the endpoints.
             # Let P_x and P_y be low-order polynomials.
             
             # Let's restart the logic with this constrained parameterization.

    # RESTART WITH CONSTRAINED PARAMETERIZATION
    
    def calculate_area_constrained(params, n_angles=180):
        """
        params: coefficients for the "free" part of the path.
        Path:
        x(t) = -2*t + t*(1-t) * (p0 + p1*t + p2*t^2)
        y(t) = 0 + t*(1-t) * (q0 + q1*t + q2*t^2)
        where t = a/90.
        """
        p = params[:3]
        q = params[3:]
        
        angles_deg = np.linspace(0.0, 90.0, n_angles)
        t = angles_deg / 90.0
        angles_rad = np.deg2rad(angles_deg)
        
        # Compute x, y
        poly_x = p[0] + p[1]*t + p[2]*t**2
        poly_y = q[0] + q[1]*t + q[2]*t**2
        
        xs_path = -2.0 * t + t * (1.0 - t) * poly_x
        ys_path = t * (1.0 - t) * poly_y
        
        inside_all = np.ones(n_points, dtype=bool)
        
        for i in range(n_angles):
            a_rad = angles_rad[i]
            x_c = xs_path[i]
            y_c = ys_path[i]
            
            dx = gx - x_c
            dy = gy - y_c
            
            c = np.cos(a_rad)
            s = np.sin(a_rad)
            
            u = dx * c + dy * s
            v = -dx * s + dy * c
            
            mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            mask = mask1 | mask2
            
            inside_all &= mask
            
            if not np.any(inside_all):
                return 0.0
                
        return np.sum(inside_all) * 0.0001

    def objective_constrained(params):
        area = calculate_area_constrained(params)
        return -area

    # Initial guess for constrained params
    # Try to match the baseline shape.
    # Baseline x: 0 -> -0.1 -> ... -> -2.
    # Baseline y: 0 -> 0.1 -> ... -> 0.
    # Let's just start with zeros (straight line) and let the optimizer find the curve.
    init_params_c = np.zeros(6)
    
    current_score_c = -objective_constrained(init_params_c)
    best_params_c = init_params_c.copy()
    
    # Optimize
    result_c = minimize(objective_constrained, init_params_c, method='Nelder-Mead', options=options)
    
    if result_c.fun < objective_constrained(init_params_c):
        new_score_c = -result_c.fun
        if new_score_c > current_score_c:
            current_score_c = new_score_c
            best_params_c = result_c.x.copy()
            
    # Try a few more restarts with different initializations
    for i in range(10):
        if time.time() > deadline - 30.0:
            break
        # Random small perturbations
        noise = np.random.normal(0, 0.2, size=6)
        p_init = best_params_c + noise
        res = minimize(objective_constrained, p_init, method='Nelder-Mead', options={'maxiter': 200})
        if res.fun < objective_constrained(best_params_c):
            best_params_c = res.x.copy()
            current_score_c = -res.fun

    # Convert best parameters back to path points for the return value
    # The evaluator expects an array of (x, y, a) for intermediate steps.
    # We should return points that, when interpolated by the evaluator, trace our optimized curve.
    # The evaluator interpolates linearly between my returned points.
    # To best approximate the smooth curve, I should return points at regular angle intervals.
    
    # How many points to return?
    # The evaluator uses m = max(1, ceil(|a2-a1|)) steps.
    # If I return points every 1 degree, m=1, no extra interpolation.
    # If I return points every 5 degrees, m=5, it interpolates.
    # Returning more points allows better approximation of the curve.
    # Let's return points at every 2 degrees from 2 to 88 (avoiding endpoints 0 and 90).
    
    angles_out = np.arange(2.0, 90.0, 2.0)
    
    t_out = angles_out / 90.0
    p = best_params_c[:3]
    q = best_params_c[3:]
    
    poly_x = p[0] + p[1]*t_out + p[2]*t_out**2
    poly_y = q[0] + q[1]*t_out + q[2]*t_out**2
    
    xs_out = -2.0 * t_out + t_out * (1.0 - t_out) * poly_x
    ys_out = t_out * (1.0 - t_out) * poly_y
    
    path_out = np.column_stack((xs_out, ys_out, angles_out))
    
    # Ensure the result is valid (finite, etc.)
    if not np.all(np.isfinite(path_out)):
        # Fallback to a simple path
        path_out = np.array([
            [-0.5, 0.0, 30.0],
            [-1.0, 0.0, 60.0]
        ])
        
    return path_out
