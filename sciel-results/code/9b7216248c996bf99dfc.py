import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path for the moving sofa problem.
    Strategy:
    1. Parameterize the path (x, y) as high-degree Fourier series functions of angle a (0 to 90).
       x(a) = -2 * (a/90) + (a/90)*(1 - a/90) * sum_{k=1}^{K} [ Ax_k * cos(k*pi*a/90) + Bx_k * sin(k*pi*a/90) ]
       y(a) = 0 * (a/90) + (a/90)*(1 - a/90) * sum_{k=1}^{K} [ Ay_k * cos(k*pi*a/90) + By_k * sin(k*pi*a/90) ]
       This ensures endpoints are fixed at (0,0,0) and (-2,0,90) and allows for complex shapes.
    2. Use a dense set of angles to evaluate the intersection area.
    3. Use a fast local optimizer (Nelder-Mead) with time bounds to refine the Fourier coefficients.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Grid setup
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = grid_points.shape[0]
    
    # Precompute grid point coordinates for broadcasting
    gx = grid_points[:, 0]
    gy = grid_points[:, 1]

    # Fourier basis setup
    K = 6 # Number of harmonics
    # Parameters: Ax_1..Ax_K, Bx_1..Bx_K, Ay_1..Ay_K, By_1..By_K
    n_params = 4 * K

    def calculate_area(params, n_angles=360):
        """
        params: array of shape (4*K,)
        """
        # Extract coefficients
        Ax = params[0:K]
        Bx = params[K:2*K]
        Ay = params[2*K:3*K]
        By = params[3*K:4*K]
        
        # Angles from 0 to 90 degrees
        angles_deg = np.linspace(0.0, 90.0, n_angles)
        angles_rad = np.deg2rad(angles_deg)
        
        # Normalize angle t in [0, 1]
        t = angles_deg / 90.0
        theta = np.pi * t # Phase for Fourier terms: k * pi * t
        
        # Precompute Fourier basis for all angles
        # shape: (n_angles, K)
        cos_terms = np.cos(np.outer(theta, np.arange(1, K+1)))
        sin_terms = np.sin(np.outer(theta, np.arange(1, K+1)))
        
        # Evaluate polynomials (Fourier series)
        # x_free(t) = sum Ax_k * cos(k*theta) + Bx_k * sin(k*theta)
        x_free = cos_terms @ Ax + sin_terms @ Bx
        y_free = cos_terms @ Ay + sin_terms @ By
        
        # Apply envelope to ensure endpoints are 0
        envelope = t * (1.0 - t)
        xs_path = -2.0 * t + envelope * x_free
        ys_path = 0.0 * t + envelope * y_free
        
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

    def objective(params):
        area = calculate_area(params)
        return -area

    # Initial guess
    # Start with a simple curve that approximates the known good baseline.
    # The baseline moves roughly linearly in x, and has a slight bump in y.
    # Let's initialize with a single sine term for y to create a bump.
    init_params = np.zeros(n_params)
    # Add a bump in y: roughly 0.5 * sin(pi * t)
    # In our Fourier basis: y_free = By_1 * sin(pi*t)
    # We want envelope * y_free to be ~ 0.5 * sin(pi*t)
    # Since envelope = t(1-t), and sin(pi*t) is 0 at ends, it's not a perfect match but close enough for init.
    # Actually, let's just start with a small random perturbation or a simple linear-ish curve.
    # Let's try to mimic the "arch" shape.
    # x should be close to -2t. So x_free should be small.
    # y should be positive in the middle.
    
    # Let's initialize By_1 (coefficient of sin(pi*t) for y) to be positive.
    # y_path = t(1-t) * By_1 * sin(pi*t)
    # Max of t(1-t) is 0.25. Max of sin is 1.
    # So max y is 0.25 * By_1.
    # If we want max y ~ 0.5, By_1 ~ 2.
    init_params[3*K] = 2.0 # Ay_1? No, Ay is cos. By is sin.
    # Indexing:
    # 0..K-1: Ax
    # K..2K-1: Bx
    # 2K..3K-1: Ay
    # 3K..4K-1: By
    # So By_1 is at index 3*K.
    init_params[3*K] = 2.0
    
    current_score = -objective(init_params)
    best_params = init_params.copy()

    # Optimization
    # Nelder-Mead is robust. We will run it in a loop with time checks.
    
    options = {
        'maxiter': 500,
        'xatol': 1e-5,
        'fatol': 1e-7,
        'adaptive': True
    }
    
    # Run initial optimization
    if time.time() < deadline - 60:
        result = minimize(objective, init_params, method='Nelder-Mead', options=options)
        if result.fun < objective(best_params):
            best_params = result.x.copy()
            current_score = -result.fun

    # Perform multiple local restarts with small perturbations
    # This helps escape local minima created by the grid discretization.
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Perturb best params
        noise_scale = 0.5
        noise = np.random.normal(0, noise_scale, size=n_params)
        perturbed_params = best_params + noise
        
        # Reduce noise scale as we go? Or keep it constant?
        # Let's keep it moderate to explore nearby basins.
        
        res = minimize(objective, perturbed_params, method='Nelder-Mead', options={
            'maxiter': 200,
            'xatol': 1e-5,
            'fatol': 1e-7,
            'adaptive': True
        })
        
        if res.fun < objective(best_params):
            best_params = res.x.copy()
            current_score = -res.fun
            
    # Convert best parameters back to path points for the return value
    # Return points at every 1 degree from 1 to 89.
    # This ensures m=1 in the evaluator's interpolation, so no extra points are added.
    angles_out = np.arange(1.0, 90.0, 1.0)
    
    t_out = angles_out / 90.0
    theta_out = np.pi * t_out
    
    cos_terms_out = np.cos(np.outer(theta_out, np.arange(1, K+1)))
    sin_terms_out = np.sin(np.outer(theta_out, np.arange(1, K+1)))
    
    Ax = best_params[0:K]
    Bx = best_params[K:2*K]
    Ay = best_params[2*K:3*K]
    By = best_params[3*K:4*K]
    
    x_free_out = cos_terms_out @ Ax + sin_terms_out @ Bx
    y_free_out = cos_terms_out @ Ay + sin_terms_out @ By
    
    envelope_out = t_out * (1.0 - t_out)
    xs_out = -2.0 * t_out + envelope_out * x_free_out
    ys_out = 0.0 * t_out + envelope_out * y_free_out
    
    path_out = np.column_stack((xs_out, ys_out, angles_out))
    
    # Ensure the result is valid (finite, etc.)
    if not np.all(np.isfinite(path_out)):
        # Fallback to a simple path
        path_out = np.array([
            [-0.5, 0.0, 30.0],
            [-1.0, 0.0, 60.0]
        ])
        
    return path_out
