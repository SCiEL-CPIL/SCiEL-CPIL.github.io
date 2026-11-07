import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes predictions using a data-driven Gaussian RBF basis.
    
    The basis functions are centered at the unique training points (x_i, y_i).
    The width (sigma) is fixed based on the scale of the input domain [-1, 1].
    The 10 parameters are weights for the 10 selected basis functions.
    
    Note: The evaluator fits 'params' using BFGS starting from all ones.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    params = np.asarray(params).flatten()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # We need exactly 10 basis functions.
    # If we have fewer than 10 points, we duplicate or handle edge case.
    # If we have more, we select a representative subset.
    
    # Strategy for selecting centers:
    # 1. If n_points <= 10, use all points. If fewer, repeat the last point to fill up to 10.
    # 2. If n_points > 10, we should select 10 well-distributed points.
    #    Using a simple grid or random sampling might be unstable for BFGS if centers are too close.
    #    However, since the inputs are uniform in [-1,1], a fixed grid of 10 points 
    #    might be more stable and "universal" than data-dependent centers which 
    #    change every time the function is called (which is bad for caching/consistency 
    #    if the evaluator calls it multiple times with slightly different subsets? 
    #    Actually, the evaluator calls it with the TRAINING set to fit, and TEST set to predict.
    #    Wait, the signature is equation(x, y, params). 
    #    During fitting, x, y are TRAINING data.
    #    During testing, x, y are TEST data.
    #    
    #    CRITICAL ISSUE: The centers must be FIXED. 
    #    If I use the training points as centers, the basis changes between the fitting phase 
    #    and the testing phase (because the input x,y to the function changes).
    #    The parameters 'params' are fitted on the training basis. 
    #    If I use different centers for the test basis, the meaning of 'params' changes, 
    #    and the prediction will be garbage.
    #    
    #    Therefore, I MUST use a FIXED set of centers that does not depend on the input x, y.
    #    
    #    Revised Strategy:
    #    Use a fixed set of 10 centers distributed over the domain [-1, 1] x [-1, 1].
    #    Since the function is unknown, a uniform grid is a reasonable prior.
    #    Let's create a 3x3 grid plus 1 center, or a 4x4 grid subset?
    #    10 points is small. 
    #    Let's use a fixed grid of 10 points.
    #    Example: 
    #    x_coords: [-0.8, -0.4, 0.0, 0.4, 0.8]
    #    y_coords: [-0.8, -0.4, 0.0, 0.4, 0.8]
    #    We need 10 pairs. 
    #    Maybe a 3x3 grid (9 points) + center? No, center is in 3x3.
    #    Let's use a 4x4 grid and pick 10? Or just hardcode 10 well-spaced points.
    
    # Fixed centers in [-1, 1] x [-1, 1]
    # Using a stratified grid approach for coverage
    # 10 points:
    # Let's try a 3x3 grid covering [-1,1] with step 1.0? No, too coarse.
    # Let's use specific coordinates.
    
    cx = np.array([
        -0.7,  0.0,  0.7,
        -0.7,  0.0,  0.7,
        -0.7,  0.0,  0.7,
         0.0
    ])
    cy = np.array([
        -0.7, -0.7, -0.7,
         0.0,  0.0,  0.0,
         0.7,  0.7,  0.7,
         0.0
    ])
    
    # This gives 10 centers.
    # Centers:
    # (-0.7, -0.7), (0, -0.7), (0.7, -0.7)
    # (-0.7, 0),    (0, 0),    (0.7, 0)
    # (-0.7, 0.7),  (0, 0.7),  (0.7, 0.7)
    # (0, 0)  <- Duplicate of center (1,1) in the 3x3? 
    # The 3x3 grid has (0,0) at index 4. 
    # So we have a duplicate center. This makes the basis linearly dependent.
    # BFGS might handle it, but it's inefficient.
    # Let's adjust to 10 unique points.
    
    # Better set of 10 unique points in [-1,1]:
    # 4 corners, 4 midpoints of edges, 1 center, 1 extra?
    # Or a 3x3 grid (9 pts) + 1 point at (0.5, 0.5)?
    
    cx = np.array([
        -1.0,  0.0,  1.0,
        -1.0,  0.0,  1.0,
        -1.0,  0.0,  1.0,
         0.5
    ])
    cy = np.array([
        -1.0, -1.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  1.0,  1.0,
         0.5
    ])
    
    # Check uniqueness:
    # (-1,-1), (0,-1), (1,-1)
    # (-1,0),  (0,0),  (1,0)
    # (-1,1),  (0,1),  (1,1)
    # (0.5, 0.5)
    # All unique. Good.
    
    # Width (sigma) for the Gaussian RBF.
    # Domain size is 2x2. 
    # A typical sigma might be around 0.5 to 1.0.
    # If sigma is too small, the basis functions are very local and nearly orthogonal, 
    # but might not capture global trends well with only 10 centers.
    # If sigma is too large, they overlap significantly.
    # Let's try sigma = 0.8.
    
    sigma = 0.8
    
    # Compute distances from input points (x, y) to each of the 10 centers.
    # x: (N,), cx: (10,) -> dist: (N, 10)
    
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # Gaussian RBF: exp(-dist_sq / (2 * sigma^2))
    # To avoid underflow/overflow issues, we can compute in log space or just let numpy handle it.
    # exp(-large) -> 0, which is fine.
    
    rbf = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # rbf shape: (N, 10)
    # params shape: (10,)
    
    # Prediction: rbf @ params
    prediction = rbf.dot(params)
    
    # Ensure finite values
    # If any params are huge, prediction might be inf. 
    # BFGS should prevent this, but let's be safe.
    if not np.all(np.isfinite(prediction)):
        # Replace non-finite with 0 or mean? 
        # The contract says non-finite costs 1e30 during fitting.
        # If this happens during test, it's invalid.
        # We can't really fix it easily without knowing the scale.
        # Just return what we have, hoping BFGS kept params reasonable.
        pass
        
    return prediction
