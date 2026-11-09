import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Gaussian Basis Function (RBF) network.
    
    Strategy:
    Instead of a fixed polynomial or trigonometric basis, we use a sum of 10 Gaussian kernels.
    The 10 parameters are interpreted as the centers of these Gaussians in the (x, v) plane.
    Specifically, params[0:10] are mapped to centers (cx_i, cy_i) for i=0..9.
    However, we have only 10 parameters total. 
    To create 10 distinct centers, we can't have 20 parameters.
    
    Revised Strategy:
    We interpret the 10 parameters as defining a specific structure.
    Let's use a "Radial Basis Function" approach where the parameters define the centers 
    of a small number of Gaussians, or we use a single Gaussian with parameters defining 
    center, width, and amplitude? No, 10 params is a lot for one.
    
    Let's try: 5 Gaussians. 
    Params 0:4 -> Centers (cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, cx5, cy5)? 
    That uses all 10 params for centers. But we need amplitudes and widths.
    
    Alternative: 
    Use a fixed set of centers (e.g., grid points in the input range) and let the 10 params 
    be the weights? 
    Input range: x in [-0.57, 0.53], v in [-0.18, 0.28].
    If we fix the centers, the function becomes linear in the parameters.
    f(x, v) = sum_{i=0}^{9} w_i * exp(-((x-cx_i)^2 + (v-cy_i)^2) / (2*sigma^2))
    
    If the basis is fixed, BFGS will solve a linear least squares problem essentially 
    (if the centers are fixed). This is extremely stable and fast.
    The question is: can 10 fixed Gaussians approximate the target?
    The target is a nonlinear oscillator response. It is smooth.
    A set of 10 well-placed Gaussians can approximate any smooth function on a compact domain 
    very well.
    
    Let's choose 10 fixed centers that cover the input domain.
    x_range: [-0.6, 0.6]
    v_range: [-0.3, 0.3]
    
    We can pick a 3x3 grid + 1 center? Or 2x5?
    Let's pick a grid of 10 points.
    x_centers = np.linspace(-0.5, 0.5, 4) -> 4 points
    v_centers = np.linspace(-0.2, 0.2, 3) -> 3 points
    Total 12 points. We only have 10 params.
    
    Let's use 10 specific points:
    (-0.5, -0.2), (-0.5, 0.0), (-0.5, 0.2)
    ( 0.0, -0.2), ( 0.0, 0.0), ( 0.0, 0.2)
    ( 0.5, -0.2), ( 0.5, 0.0), ( 0.5, 0.2)
    ( 0.0, 0.0) is duplicate? No, let's just define 10 unique points.
    
    Actually, if the basis is fixed, the parameters are just weights. 
    The evaluator fits `params` (length 10). 
    If I define the equation as a linear combination of 10 fixed basis functions, 
    the optimization problem is convex (linear in params). BFGS will find the global minimum 
    of the MSE with respect to the weights very quickly.
    
    This is a "Linear RBF" model.
    
    Basis functions:
    phi_i(x, v) = exp(-((x - cx_i)^2 + (v - cy_i)^2) / (2 * sigma^2))
    
    We need to choose sigma. 
    If sigma is too small, the basis is over-localized and may not interpolate well if centers are sparse.
    If sigma is too large, the basis functions are nearly identical, leading to ill-conditioning.
    
    Input domain size is roughly 1.1 in x and 0.5 in v.
    A reasonable sigma is around 0.3 to 0.5.
    Let's fix sigma = 0.4.
    
    Centers:
    Let's distribute them evenly.
    x_vals = [-0.5, -0.2, 0.2, 0.5]
    v_vals = [-0.2, 0.0, 0.2]
    
    Grid:
    1. (-0.5, -0.2)
    2. (-0.5,  0.0)
    3. (-0.5,  0.2)
    4. (-0.2, -0.2)
    5. (-0.2,  0.0)
    6. (-0.2,  0.2)
    7. ( 0.2, -0.2)
    8. ( 0.2,  0.0)
    9. ( 0.2,  0.2)
    10. ( 0.5, 0.0)  # Add one more to cover the right side better?
    
    Or simply a 2x5 grid?
    x = [-0.5, 0.5]
    v = [-0.2, -0.1, 0.0, 0.1, 0.2]
    This gives 10 points.
    
    Let's verify the coverage.
    x in [-0.57, 0.53]. The grid covers [-0.5, 0.5]. This is good.
    v in [-0.18, 0.28]. The grid covers [-0.2, 0.2]. This is good.
    
    The function will be:
    y_pred = sum_{i=0}^{9} p_i * exp(-((x - cx_i)^2 + (v - cy_i)^2) / (2 * 0.4^2))
    
    This is linear in p. BFGS will solve it effectively.
    The "fit" step in the evaluator minimizes MSE. Since the model is linear in params, 
    the optimal params are the solution to the linear least squares problem.
    BFGS is an iterative method, but for a quadratic objective (MSE of linear model), it converges 
    quickly.
    
    Why might this be better?
    1. Avoids the "Duffing" structural bias which might be wrong if the oscillator isn't exactly Duffing.
    2. Universal approximator capability (with enough Gaussians).
    3. Smoothness: Gaussians are C-infinity, matching the smoothness of physical oscillator states.
    4. Stability: Fixed basis functions prevent parameter drift into regions where the function becomes unstable or undefined (unlike rational functions).
    
    Let's implement this.
    """
    
    # Define fixed centers for the 10 Gaussian basis functions
    # 2 x-values, 5 y-values
    cx_vals = np.array([-0.5, 0.5])
    cy_vals = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    
    # Create meshgrid for centers
    cx_grid, cy_grid = np.meshgrid(cx_vals, cy_vals, indexing='ij')
    cx_flat = cx_grid.flatten()
    cy_flat = cy_grid.flatten()
    
    # Ensure we have exactly 10 centers
    # 2 * 5 = 10. Correct.
    
    # Define bandwidth (sigma)
    # Input ranges: x ~ 1.1, v ~ 0.5.
    # A sigma of 0.3-0.5 seems appropriate to allow overlap but maintain locality.
    sigma = 0.4
    sigma2_2 = 2 * sigma * sigma
    
    # Calculate distances from (x, v) to each center
    # x shape: (N,)
    # cx_flat shape: (10,)
    # We need (N, 10)
    
    dx = x[:, np.newaxis] - cx_flat[np.newaxis, :]
    dv = v[:, np.newaxis] - cy_flat[np.newaxis, :]
    
    # Squared distances
    dist_sq = dx**2 + dv**2
    
    # Gaussian basis values
    # Shape: (N, 10)
    basis = np.exp(-dist_sq / sigma2_2)
    
    # Linear combination with params
    # params shape: (10,)
    # result shape: (N,)
    result = basis @ params
    
    return result
