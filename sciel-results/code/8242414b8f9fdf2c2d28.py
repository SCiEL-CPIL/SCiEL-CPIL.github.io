import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

def compute_objective(params):
    # params: 3*N array: x1, y1, th1, x2, y2, th2, ...
    x = params[0::3]
    y = params[1::3]
    th = params[2::3]
    
    # Compute vertices for all hexagons
    # Shape: (N, 6, 2)
    angles = np.arange(6) * np.pi / 3.0
    # cos and sin of (th_i + angle_j)
    # x_i + cos(th_i + angle_j)
    # y_i + sin(th_i + angle_j)
    
    # Efficient vectorization
    # th: (N, 1), angles: (1, 6)
    th_exp = th[:, np.newaxis]
    ang_exp = angles[np.newaxis, :]
    
    cos_terms = np.cos(th_exp + ang_exp)
    sin_terms = np.sin(th_exp + ang_exp)
    
    vx = x[:, np.newaxis] + cos_terms
    vy = y[:, np.newaxis] + sin_terms
    
    # vx, vy are (N, 6)
    
    # Compute max projection for outer hexagon (orientation 0)
    # Normals n_k for k=0..5
    n_angles = np.arange(6) * np.pi / 3.0
    nx = np.cos(n_angles)
    ny = np.sin(n_angles)
    
    # Projection of each vertex v on each normal n_k
    # v: (N, 6, 2) -> reshape to (N*6, 2)
    v_flat = np.column_stack((vx.ravel(), vy.ravel())) # (72, 2)
    
    # Projections: (72, 6)
    proj = v_flat[:, 0] * nx[np.newaxis, :] + v_flat[:, 1] * ny[np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    
    S = max_proj * 2.0 / np.sqrt(3.0)
    
    # Compute overlap penalty
    # Distance between centers
    # x, y: (N,)
    # dist^2: (N, N)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d_sq = dx**2 + dy**2
    d = np.sqrt(d_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), 1)
    d_pairs = d[mask]
    
    # Penalty: sum of (max(0, sqrt(3) - d))^2
    # Min distance to avoid overlap is sqrt(3) (inradius * 2)
    min_d = np.sqrt(3.0)
    overlaps = np.maximum(0, min_d - d_pairs)
    penalty = np.sum(overlaps**2)
    
    return S + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Bounds
    # x, y in [-5, 5]
    # theta in [0, pi/3]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, np.pi/3.0)) # theta
        
    # Initial guess: Grid
    # 4x3 grid?
    # Spacing ~1.8
    init_x = []
    init_y = []
    init_th = []
    # 4 columns, 3 rows
    # x: -2.7, -0.9, 0.9, 2.7
    # y: -1.8, 0, 1.8
    # Offset for hexagonal packing?
    # Let's just use a simple grid for DE initial population, DE doesn't need init.
    
    # Run Differential Evolution
    best_result = None
    try:
        # DE is stochastic, set seed for reproducibility? No, random is fine.
        # maxiter and popsize control time.
        # 36 dimensions.
        # popsize=15 (default is 15*dim? No, default is 15. For 36 dims, 15 is small. Let's use 20).
        # maxiter: start with 50, increase if time permits.
        
        # We will run DE in chunks to check time.
        
        # Strategy: Run DE with a timeout.
        # scipy DE doesn't have a built-in timeout.
        # We can use a callback to check time?
        # Callback receives (xk, fk, convergence).
        
        def callback(xk, fk, convergence):
            if time.time() > deadline - 20.0: # Leave 20s for polish
                raise TimeoutError("DE Time Limit")
            return False
        
        try:
            res = differential_evolution(
                compute_objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-8, 
                seed=42,
                callback=callback,
                polish=False # We will do our own polish
            )
            best_result = res
        except TimeoutError:
            # If timeout, we need the best result found so far.
            # Unfortunately, differential_evolution doesn't return intermediate best on exception easily.
            # Alternative: Run DE manually or use a smaller budget.
            # Let's assume DE finishes within 20s usually for 36 dims?
            # 36 dims * 20 pop * 100 iter = 72000 function evals.
            # Each eval is fast (numpy). 72k evals should be < 5s.
            # So timeout is unlikely unless machine is very slow.
            pass

        if best_result is None or not best_result.success and time.time() > deadline - 30:
             # Fallback to a simple grid search if DE fails or takes too long?
             # Or just use a random start.
             pass

        # Polish the best result
        x0 = best_result.x
        # Use Nelder-Mead for final polish
        res_polish = minimize(
            compute_objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        best_x = res_polish.x
        best_val = res_polish.fun
        
    except Exception as e:
        # Fallback: Return a known valid configuration if optimizer fails
        # Simple grid
        best_x = np.zeros(3*N)
        # Place in a line? No, grid.
        for i in range(N):
            best_x[i*3] = (i % 4 - 1.5) * 1.8
            best_x[i*3+1] = (i // 4 - 1) * 1.8
            best_x[i*3+2] = 0.0
        best_val = compute_objective(best_x)

    # Extract parameters
    x = best_x[0::3]
    y = best_x[1::3]
    th = best_x[2::3]
    
    # Calculate final S (without penalty for return value? 
    # The score is the outer side length. The penalty was for optimization.
    # We should return the actual geometric S, assuming overlaps are resolved.
    # If overlaps remain, the configuration is invalid.
    # We should check if penalty is 0.
    # If penalty > 0, we might have an invalid solution.
    # However, the evaluator checks validity? "invalid result ... scores 0".
    # We must ensure disjointness.
    
    # Recalculate S from best_x
    S_val = compute_objective(best_x) # This includes penalty.
    # We need the pure S.
    # Let's make a helper to get pure S.
    
    def get_pure_S(params):
        x = params[0::3]
        y = params[1::3]
        th = params[2::3]
        angles = np.arange(6) * np.pi / 3.0
        th_exp = th[:, np.newaxis]
        ang_exp = angles[np.newaxis, :]
        vx = x[:, np.newaxis] + np.cos(th_exp + ang_exp)
        vy = y[:, np.newaxis] + np.sin(th_exp + ang_exp)
        v_flat = np.column_stack((vx.ravel(), vy.ravel()))
        n_angles = np.arange(6) * np.pi / 3.0
        nx = np.cos(n_angles)
        ny = np.sin(n_angles)
        proj = v_flat[:, 0] * nx[np.newaxis, :] + v_flat[:, 1] * ny[np.newaxis, :]
        max_proj = np.max(np.abs(proj))
        return max_proj * 2.0 / np.sqrt(3.0)
        
    pure_S = get_pure_S(best_x)
    
    # Check for overlaps
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d = np.sqrt(dx**2 + dy**2)
    mask = np.triu(np.ones((N, N), dtype=bool), 1)
    d_pairs = d[mask]
    min_d = np.sqrt(3.0)
    overlaps = np.maximum(0, min_d - d_pairs)
    
    if np.sum(overlaps**2) > 1e-6:
        # Invalid solution. Try a safe fallback.
        # This shouldn't happen if penalty was high enough.
        # Fallback: Spread them out more.
        # Just return the grid.
        inner = []
        for i in range(N):
            inner.append([(i % 4 - 1.5) * 2.0, (i // 4 - 1) * 2.0, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([float(x[i]), float(y[i]), float(np.degrees(th[i]))])
        
    return inner, (0.0, 0.0), float(pure_S), 0.0

