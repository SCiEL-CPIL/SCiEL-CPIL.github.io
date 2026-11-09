import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 615.0  # Safety margin before 630s kill

    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # Pre-compute base corridor mask for the grid if we were at (0,0,0)?
    # No, the corridor moves. We need to check each point against the corridor.
    
    # To speed up evaluation, we can pre-compute the "active" points.
    # A point is active if it is inside the current intersection.
    
    def evaluate_path_fast(path, active_mask=None):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        active_mask: optional boolean mask of points to check (from previous step)
        Returns: (score, new_active_mask)
        """
        # Ensure path is a numpy array
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            
        # Prepend and append fixed endpoints
        # The evaluator puts (0, 0, 0) before and (-2, 0, 90) after.
        # Our path contains intermediate points.
        # We need to simulate the evaluator's densification.
        
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        if active_mask is None:
            valid = np.ones(n_points, dtype=bool)
        else:
            valid = active_mask.copy()
            
        # If no points are valid, return 0
        if not np.any(valid):
            return 0.0, valid
            
        # We only check points that are currently valid
        # To be efficient, we can just check all points but mask the result?
        # No, checking 135k points * 90 angles is expensive.
        # Checking only valid points is better if the set shrinks.
        
        # However, vectorizing over a subset of P is tricky if we want to update the mask.
        # Let's just vectorize over all P but early exit if the count drops too low?
        # Actually, the current program checks all P for all angles.
        # Let's try to optimize the check itself.
        
        # For each placement, compute u, v for all P, then mask.
        # This is O(N * M) where N is points, M is placements.
        
        # Let's use the active mask to restrict computation if possible?
        # P[valid] is a copy. We can compute u, v for P[valid].
        # But then we need to map back to the full mask.
        
        # Let's stick to checking all points but use NumPy efficiently.
        # The bottleneck is the 90 iterations of trig and comparisons.
        
        # Optimization: Pre-compute cos/sin for all angles in dense_path?
        # dense_path is small (~90-100 points).
        
        # Let's just loop through dense_path and update valid.
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor:
            # Arm 1: 0 <= u <= 1, -3 <= v <= 1
            # Arm 2: -3 <= u <= 1, 0 <= v <= 1
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            # Early exit if too few points remain
            count = np.sum(valid)
            if count < 100: # Threshold to avoid wasting time on tiny areas
                return 0.0, valid
                
        score = np.sum(valid) * 0.0001
        return score, valid

    # Strategy:
    # Use a Fourier series to represent the path x(a) and y(a).
    # This reduces the dimensionality from ~178 to ~2*K.
    # We will optimize the coefficients of the Fourier series.
    
    # Define the number of harmonics. 
    # K=5 gives 10 parameters (5 cos + 5 sin for x, 5 cos + 5 sin for y? No, just coefficients).
    # Let's use a simple trigonometric polynomial.
    # x(a) = c_x0 + sum(c_xk * cos(k*pi*a/90) + s_xk * sin(k*pi*a/90))
    # y(a) = c_y0 + sum(c_yk * cos(k*pi*a/90) + s_yk * sin(k*pi*a/90))
    
    K = 4 # Number of harmonics. Total params = 2 * (2*K + 1) = 20 params.
    n_params = 2 * (2 * K + 1)
    
    # Initial guess:
    # x(a) should go from 0 to -2.
    # y(a) should bump up.
    # Let's initialize the Fourier coefficients to approximate a linear x and a sinusoidal y.
    
    # For x(a) = -2 * a/90 = -2/90 * a.
    # a = 90 * t, t in [0,1]. x = -2t.
    # t is linear in a. The first harmonic sin(pi*a/90) is roughly linear near 0? No.
    # It's easier to just start with a known good path and fit Fourier coeffs?
    # Or just start with zeros and let optimizer find it?
    # Let's start with a rough fit.
    
    # Let's just use the sinusoidal bump as the initial path and extract Fourier coeffs?
    # That's complex. Let's just start with a simple parameterization and let NM do the work.
    # Initial params:
    # x0 = -1 (average x)
    # y0 = 0 (average y)
    # Other coeffs = 0.
    
    # Actually, let's construct the initial path explicitly and use it to seed the optimizer?
    # No, NM needs a vector of parameters.
    
    # Let's define the function to generate the path from parameters.
    def generate_path_from_coeffs(params):
        # params: [x0, x1c, x1s, x2c, x2s, ..., y0, y1c, y1s, ...]
        # a in degrees 1..89
        a = np.arange(1, 90, dtype=float)
        t = a / 90.0
        
        x = np.zeros_like(a)
        y = np.zeros_like(a)
        
        # x coefficients
        x0 = params[0]
        x[0:] = x0
        for k in range(1, K+1):
            idx_c = 1 + (k-1)*2
            idx_s = 2 + (k-1)*2
            x += params[idx_c] * np.cos(k * np.pi * t)
            x += params[idx_s] * np.sin(k * np.pi * t)
            
        # y coefficients
        y0 = params[n_params // 2]
        y[0:] = y0
        for k in range(1, K+1):
            idx_c = n_params // 2 + 1 + (k-1)*2
            idx_s = n_params // 2 + 2 + (k-1)*2
            y += params[idx_c] * np.cos(k * np.pi * t)
            y += params[idx_s] * np.sin(k * np.pi * t)
            
        return np.column_stack([x, y, a])
    
    # Objective function for scipy.optimize
    def objective(params):
        path = generate_path_from_coeffs(params)
        score, _ = evaluate_path_fast(path)
        # We want to maximize score, so return negative
        return -score
    
    # Initial parameters
    # Let's try to match the linear x and bump y roughly.
    # x(a) = -2 * a/90. 
    # This is not periodic, so Fourier series will have Gibbs phenomena at boundaries if we don't handle it.
    # However, the path is only evaluated at 1..89.
    # A better initial guess might be to fit the linear function with the available harmonics.
    # Or just start with zeros and let NM move it.
    # Given NM is a local method, a good start is crucial.
    
    # Let's use a known good path to initialize the parameters by least squares?
    # That might be overkill. Let's just start with a simple guess.
    # x0 = -1.0 (midpoint)
    # The linear term is hard to represent with cos/sin on [0,1] without DC offset.
    # Let's just start with all zeros except x0=-1, y0=0.5 (guess for bump height).
    
    initial_params = np.zeros(n_params)
    initial_params[0] = -1.0 # x0
    initial_params[n_params // 2] = 0.5 # y0
    
    # Run Nelder-Mead
    best_result = None
    best_score = -np.inf
    
    # We might need multiple restarts or a larger maxiter
    # NM can get stuck.
    
    # Let's try a few restarts with different initial guesses if time permits.
    # Restart 1: The guess above.
    
    res = minimize(objective, initial_params, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if res.success or res.fun < -best_score:
        best_score = -res.fun
        best_params = res.x
        best_result = generate_path_from_coeffs(best_params)
        
    # Check time
    if time.time() > deadline - 30:
        return best_result if best_result is not None else generate_path_from_coeffs(initial_params)
        
    # Restart 2: Different initial guess?
    # Maybe a higher bump?
    initial_params2 = np.zeros(n_params)
    initial_params2[0] = -1.0
    initial_params2[n_params // 2] = 1.0
    
    res2 = minimize(objective, initial_params2, method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if res2.success or res2.fun < -best_score:
        best_score = -res2.fun
        best_params = res2.x
        best_result = generate_path_from_coeffs(best_params)
        
    # Final evaluation
    if best_result is None:
        best_result = generate_path_from_coeffs(initial_params)
        
    # Ensure output format is correct
    return best_result
