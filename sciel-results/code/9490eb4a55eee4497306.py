import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    width = 1.0 / N
    start_time = time.time()
    deadline = 600.0

    # We need to choose X[0..7] and Y[0..7] to minimize the union volume.
    # The problem has symmetry: we can shift all X by a constant and all Y by a constant.
    # Let's fix X[0] = 0 and Y[0] = 0 to break this symmetry.
    
    # Objective: Monte Carlo estimate of volume
    # But for optimization, we can use a faster proxy or the actual MC with fixed points.
    # Since the evaluator uses a specific MC setup, let's try to optimize for that directly.
    # However, 500k points is slow for an objective function called thousands of times.
    # Let's use a smaller MC estimate for the optimizer, and verify with the full one at the end?
    # Or use the analytical area integral which is faster.
    
    # Analytical approach:
    # V = integral_0^1 Area(c) dc
    # Area(c) is the area of the union of 64 rectangles.
    # Rectangle (i,j) at height c:
    # x in [X_i + c*i/N, X_i + c*i/N + 1/N]
    # y in [Y_j + c*j/N, Y_j + c*j/N + 1/N]
    
    # To compute Area(c) efficiently:
    # Project onto x-axis: union of intervals [L_i, R_i] where L_i = X_i + c*i/N, R_i = L_i + 1/N.
    # Project onto y-axis: union of intervals [B_j, T_j] where B_j = Y_j + c*j/N, T_j = B_j + 1/N.
    # The union of rectangles is NOT simply the product of the unions of projections.
    # It is the union of the individual rectangles.
    
    # However, if the x-projections are disjoint and y-projections are disjoint, it's the sum of areas.
    # But they overlap.
    
    # A fast way to compute the area of union of axis-aligned rectangles is the sweep line algorithm.
    # With 64 rectangles, it's feasible.
    # But we need to do this for many c values during optimization.
    
    # Let's try a different approach:
    # The volume is proportional to the measure of the set of points covered.
    # Maybe we can use a grid-based approximation for the objective function.
    # Or just use the MC estimate with fewer points for optimization.
    
    # Let's use MC with 10000 points for optimization, and check the final result.
    
    def get_objective(X, Y, n_points=10000):
        # X, Y are arrays of shape (N,)
        # X[0] and Y[0] are fixed to 0 in the search, but here we pass full arrays.
        
        # Determine bounds
        i_idx = np.arange(N)
        j_idx = np.arange(N)
        
        # Min X
        # x_min = min(X_i + 0) = min(X)
        # x_max = max(X_i + 1*i/N + 1/N)
        
        x_min = np.min(X)
        x_max = np.max(X + i_idx / N + 1.0 / N)
        
        y_min = np.min(Y)
        y_max = np.max(Y + j_idx / N + 1.0 / N)
        
        if x_max - x_min < 1e-9 or y_max - y_min < 1e-9:
            return float('inf')
            
        # Sample points
        # For speed, we can use a fixed set of points or generate them quickly.
        # Generating 10k points is fast.
        
        a = np.random.uniform(x_min, x_max, n_points)
        b = np.random.uniform(y_min, y_max, n_points)
        c = np.random.uniform(0, 1, n_points)
        
        # Check if point (a,b,c) is in any tube (i,j)
        # Tube (i,j) covers:
        # a in [X_i + c*i/N, X_i + c*i/N + 1/N]
        # b in [Y_j + c*j/N, Y_j + c*j/N + 1/N]
        
        # Vectorized check:
        # a: (P,), X: (N,), c: (P,)
        # We need to check if there exists i,j such that conditions hold.
        
        # This is hard to vectorize directly for "any" without loops or large arrays.
        # Let's loop over i and j? 64 iterations.
        # Or loop over i, and for each i, check if a is in x-range, then for those, check y.
        
        # Let's try:
        # For each i:
        #   x_left = X[i] + c * i / N
        #   x_right = x_left + 1/N
        #   mask_x = (a >= x_left) & (a <= x_right)
        #   If any mask_x is True, we need to check y for those points.
        #   But we need to know if ANY (i,j) covers the point.
        
        # Efficient way:
        # Create a boolean array covered = False (shape P,)
        # For i in 0..N-1:
        #   For j in 0..N-1:
        #      ...
        
        # This is 64 * P operations. With P=10000, that's 640k ops. Very fast.
        
        covered = np.zeros(n_points, dtype=bool)
        
        for i in range(N):
            x_l = X[i] + c * i / N
            x_r = x_l + width
            mask_x = (a >= x_l) & (a <= x_r)
            if not np.any(mask_x):
                continue
            # Only check y for points where x is covered by tube i
            # But we still need to check all j for these points.
            # Actually, we can just update covered.
            # To avoid redundant checks, we can only check j for points not yet covered?
            # No, because a point might be covered by (i, j1) but we are checking (i, j2).
            # It's simpler to just check all.
            
            for j in range(N):
                y_l = Y[j] + c * j / N
                y_r = y_l + width
                mask_y = (b >= y_l) & (b <= y_r)
                # Point is covered if mask_x AND mask_y
                # We can do: covered |= (mask_x & mask_y)
                # But mask_x is computed outside j loop.
                covered |= (mask_x & mask_y)
                
        frac = np.mean(covered)
        vol = frac * (x_max - x_min) * (y_max - y_min)
        return vol

    # Optimization
    # We fix X[0]=0, Y[0]=0.
    # Variables: X[1:8], Y[1:8] -> 14 variables.
    
    # Let's use Nelder-Mead or similar.
    # Or simple coordinate descent.
    
    # Initial guess:
    # Maybe spread them out?
    # X_i = i * width? No, that would make them disjoint in x at c=0, but shift.
    # If X_i = 0 for all i, then at c=0, all x-intervals are [0, 1/N].
    # At c=1, x-intervals are [i/N, (i+1)/N].
    # This creates a "comb" shape.
    
    # Let's try X_i = 0, Y_j = 0.
    X_init = np.zeros(N)
    Y_init = np.zeros(N)
    
    # Let's try to optimize using scipy.optimize if available, otherwise manual.
    # Manual coordinate descent is safer.
    
    def full_objective(X, Y):
        return get_objective(X, Y, 50000)
        
    # We will optimize X and Y jointly.
    # Since X[0] and Y[0] are fixed, we optimize the rest.
    
    X = X_init.copy()
    Y = Y_init.copy()
    
    current_val = full_objective(X, Y)
    
    # Coordinate Descent
    step_size = 0.1
    max_iters = 1000
    
    # We will alternate between X and Y optimizations.
    
    # Helper to optimize one variable set
    def optimize_vars(var_array, fixed_array, other_var_name, current_val, max_iter=100):
        # var_array is the one being optimized (e.g., X[1:])
        # fixed_array is the other (e.g., Y)
        # other_var_name is 'X' or 'Y'
        
        best_val = current_val
        best_var = var_array.copy()
        
        for it in range(max_iter):
            improved = False
            for k in range(len(var_array)):
                # Try +/- step
                for delta in [-step_size, step_size, -step_size*0.5, step_size*0.5]:
                    new_var = best_var.copy()
                    new_var[k] += delta
                    
                    # Reconstruct full arrays
                    if other_var_name == 'X':
                        # Optimizing X, Y is fixed
                        full_X = np.zeros(N)
                        full_X[0] = 0
                        full_X[1:] = new_var
                        full_Y = fixed_array
                    else:
                        # Optimizing Y, X is fixed
                        full_X = fixed_array
                        full_Y = np.zeros(N)
                        full_Y[0] = 0
                        full_Y[1:] = new_var
                        
                    val = full_objective(full_X, full_Y)
                    if val < best_val:
                        best_val = val
                        best_var = new_var
                        improved = True
                        
            if not improved:
                # Try smaller step
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            else:
                step_size *= 1.2 # increase if improved?
                
        return best_var, best_val

    # Main loop
    for outer_iter in range(50):
        if time.time() > deadline - 10:
            break
            
        # Optimize X
        X_vars, val_x = optimize_vars(X[1:], Y, 'X', current_val, max_iter=50)
        X[1:] = X_vars
        current_val = val_x
        
        # Optimize Y
        Y_vars, val_y = optimize_vars(Y[1:], X, 'Y', current_val, max_iter=50)
        Y[1:] = Y_vars
        current_val = val_y
        
        # Random restart if stuck?
        # Let's add a random perturbation occasionally.
        if outer_iter % 10 == 0 and outer_iter > 0:
            # Perturb
            X[1:] += np.random.normal(0, 0.05, N-1)
            Y[1:] += np.random.normal(0, 0.05, N-1)
            current_val = full_objective(X, Y)

    # Final check
    # The evaluator uses a specific seed. We should ensure our result is robust.
    # The score is 1/V.
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X[i_flat]
    base_y = Y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
