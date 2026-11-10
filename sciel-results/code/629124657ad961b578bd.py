import numpy as np
import time
from scipy.optimize import differential_evolution

def fast_volume_estimate(positions, n_samples=20000):
    """
    Fast vectorized Monte Carlo estimate of the union volume.
    positions: shape (64, 2)
    """
    if not np.all(np.isfinite(positions)):
        return 1e10
        
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Determine bounding box
    # Tube k is (i, j) where i = k//8, j = k%8
    i_idx = np.arange(64) // 8
    j_idx = np.arange(64) % 8
    
    # Max extent in x: x_k + c*i/8 + 1/8. Since c in [0,1], max is x_k + i/8 + 1/8
    # Min extent in x: x_k. Min is min(x_k)
    # However, the box in the task is defined as:
    # X0 = min_k x_k
    # X1 = max_k (x_k + (i+1)/8)  <-- Wait, the task says (i+1)/8? 
    # Let's re-read carefully: "X1 = max_k (x_k + (i+1)/8)"
    # Actually, the tube is [x_k + c*i/8, x_k + c*i/8 + 1/8].
    # At c=1, right edge is x_k + i/8 + 1/8 = x_k + (i+1)/8.
    # At c=0, left edge is x_k.
    # So X1 = max_k (x_k + (i+1)/8) is correct for the bounding box of the union.
    # Similarly Y1 = max_k (y_k + (j+1)/8).
    
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / 8.0)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / 8.0)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 1e10
        
    # Sample points
    # We need to be careful with the RNG state if we want reproducibility for the same positions,
    # but for optimization, we can use a fixed seed or just rely on the fact that 
    # the evaluator uses a specific seed. 
    # However, the objective function here is just a guide.
    # To be consistent with the evaluator's logic (which uses a specific seed 12345),
    # we should ideally use the same sampling points if possible, but since the box changes,
    # the normalized coordinates would change.
    # The evaluator does:
    # a = rng.uniform(X0, X1, 500000)
    # b = rng.uniform(Y0, Y1, 500000)
    # c = rng.uniform(0, 1, 500000)
    
    # For speed, we use fewer samples here.
    # We will generate random numbers in [0,1] and map them.
    # To make the estimate stable across different box sizes, we can use the same
    # underlying random numbers for the relative positions within the box.
    
    # Let's use a fixed set of random numbers for the optimization objective to ensure
    # consistency (pseudo-deterministic objective).
    # But wait, if the box changes, the "same" relative points map to different absolute points.
    # This is actually fine and often better for optimization as it reduces variance 
    # caused by changing sample locations unrelated to the shape.
    
    # Generate randoms
    # We'll use a fixed seed for the optimizer's internal evaluation to be consistent.
    # However, the final score is computed by the evaluator with its own RNG.
    # So this is just an approximation.
    
    # To be fast, we use 20,000 samples.
    # We need to check if each point is inside ANY tube.
    # Point (a, b, c) is in tube k if:
    # x_k + c*i_k/8 <= a <= x_k + c*i_k/8 + 1/8
    # AND
    # y_k + c*j_k/8 <= b <= y_k + c*j_k/8 + 1/8
    
    # Vectorized approach:
    # Let's create arrays of shape (n_samples, 64) or (64, n_samples).
    # n_samples = 20000, 64 tubes. 20000*64 = 1.28M points. This is small for NumPy.
    
    # We'll use a fixed random state for the optimizer to make the objective smooth-ish.
    # But actually, using a different seed every time might help escape noise?
    # No, for optimization, a fixed set of sample points (control variates) is better.
    
    # Let's pre-generate the random numbers for c, and relative a, b.
    # Wait, the box changes. 
    # Let u, v, w be uniform in [0,1].
    # a = X0 + u * (X1 - X0)
    # b = Y0 + v * (Y1 - Y0)
    # c = w
    
    # We need to evaluate for all 64 tubes.
    # Let's use broadcasting.
    # u, v, w shape: (n_samples,)
    # x_k, y_k, i_k, j_k shape: (64,)
    
    # a_k = x_k + c * i_k / 8
    # b_k = y_k + c * j_k / 8
    
    # Condition: a_k <= a <= a_k + 1/8
    # a - a_k >= 0 AND a - a_k <= 1/8
    
    # Let's do it in chunks or just full broadcast if memory allows.
    # 20000 * 64 floats is ~ 10MB. Very safe.
    
    # We need to be careful: the evaluator uses a specific sequence of RNG calls.
    # rng.uniform(X0, X1, 500000)
    # rng.uniform(Y0, Y1, 500000)
    # rng.uniform(0, 1, 500000)
    
    # If we want to mimic the evaluator exactly for the final check, we'd need to replicate that.
    # But for optimization, a consistent Monte Carlo estimator is sufficient.
    
    # Let's use a fixed seed for the optimizer's internal RNG to ensure the objective function
    # is deterministic for a given set of positions.
    rng_opt = np.random.default_rng(123)
    
    # Actually, generating 20k randoms every call is cheap.
    # But to reduce noise, using the SAME randoms for every evaluation is better.
    # So we define the randoms outside the function or as a global.
    
    global _opt_rng
    if '_opt_rng' not in globals():
        _opt_rng = np.random.default_rng(999)
    
    # We need to generate new randoms? No, if we use the same randoms u,v,w,
    # the points move as the box moves. This is a valid estimator.
    # Let's generate them once and reuse?
    # If we reuse, the variance is lower.
    
    # Let's generate 20000 samples of u, v, w in [0,1]
    # We will store them globally.
    
    if '_opt_samples' not in globals():
        _opt_samples = _opt_rng.random((20000, 3))
        
    u = _opt_samples[:, 0]
    v = _opt_samples[:, 1]
    w = _opt_samples[:, 2]
    
    a = X0 + u * (X1 - X0)
    b = Y0 + v * (Y1 - Y0)
    c = w
    
    # Now check containment
    # a shape: (20000,)
    # x shape: (64,)
    
    # a_k = x_k + c * i_k / 8
    # We need to check if a is in [a_k, a_k + 1/8] for any k.
    
    # Let's expand:
    # a[:, None] shape (20000, 1)
    # x[None, :] shape (1, 64)
    # c[:, None] shape (20000, 1)
    # i_idx[None, :] shape (1, 64)
    
    # a_k = x[None, :] + c[:, None] * i_idx[None, :] / 8.0
    # This creates a (20000, 64) array.
    
    a_k = x[None, :] + c[:, None] * i_idx[None, :] / 8.0
    b_k = y[None, :] + c[:, None] * j_idx[None, :] / 8.0
    
    # Check conditions
    # (a >= a_k) & (a <= a_k + 1/8)
    # (b >= b_k) & (b <= b_k + 1/8)
    
    # To save memory, we can do it in two steps or just use the full array.
    # 20000 * 64 is small.
    
    in_x = (a[:, None] >= a_k) & (a[:, None] <= a_k + 1.0/8.0)
    in_y = (b[:, None] >= b_k) & (b[:, None] <= b_k + 1.0/8.0)
    
    in_tube = in_x & in_y
    
    # A point is in the union if it is in ANY tube.
    # in_tube shape (20000, 64). Any along axis 1.
    in_union = np.any(in_tube, axis=1)
    
    fraction = np.mean(in_union)
    
    volume = fraction * box_area
    
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # Safety margin
    
    # Initial guess: The 1D optimal solution is a good starting point.
    # Let's reconstruct the Keich positions or just use a random start for DE.
    # DE is global, so random start is fine, but a good start helps.
    
    # Let's use the 1D optimal positions as the center of the DE bounds?
    # Or just define bounds.
    
    # The positions can be anywhere. 
    # To make DE efficient, we should normalize the problem.
    # However, the volume is translation invariant? 
    # No, the bounding box changes with translation.
    # But the relative geometry is what matters for the union volume 
    # IF the box was fixed. But the box adapts to the positions.
    # Actually, the volume of the union is translation invariant in 3D space 
    # if the box was infinite, but here the box is tight.
    # Wait, V = fraction * BoxArea.
    # If we shift all x by dx, X0 and X1 shift by dx, BoxArea unchanged.
    # The points a, b also shift? 
    # a = rng.uniform(X0, X1). If X0, X1 shift, the distribution of a shifts.
    # But the relative positions of points to tubes remain the same.
    # So the fraction is invariant to global translation.
    # Thus, we can fix one coordinate to break symmetry, e.g., min(x) = 0.
    
    # However, DE works on the raw parameters.
    # Let's define bounds. A reasonable range is [-2, 3].
    # 64 variables.
    
    # To speed up DE, we can use the 1D solution as the initial population center.
    
    # Reconstruct 1D Keich positions
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Create initial 2D positions from 1D
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    init_x = keich[i_flat]
    init_y = keich[j_flat]
    
    # Add some noise to init
    init_pos = np.column_stack([init_x, init_y])
    
    # Define bounds for DE
    # We want to allow the optimizer to find the best relative positions.
    # Since translation doesn't matter, we can constrain the center or min.
    # But standard DE doesn't support equality constraints easily without penalties.
    # Let's just use a wide box.
    
    # A tighter box helps DE converge.
    # The spread of the 1D solution is about 1.
    # Let's use bounds [-1, 2] for each coordinate.
    
    bounds = [(-1.0, 2.0)] * 128
    
    # DE Parameters
    # maxiter: We have ~600s. Each function eval takes ~5-10ms (20k samples).
    # Population size: 15 * 128 = 1920 individuals.
    # 1 iteration = 1920 evals * 10ms = ~20s.
    # So we can do ~30 iterations.
    
    # Let's try to be smarter.
    # Use a smaller population?
    # Or reduce samples per eval?
    
    # Let's try n_samples = 10000 for speed.
    global _opt_samples
    _opt_samples = np.random.default_rng(999).random((10000, 3))
    
    def obj_func(flat_pos):
        pos = flat_pos.reshape(64, 2)
        return fast_volume_estimate(pos, n_samples=10000)
        
    # We will run DE in a loop, checking time.
    # DE in SciPy doesn't have a built-in time limit, only maxiter.
    # We can estimate time per iteration and set maxiter accordingly.
    
    # Let's do a quick timing test.
    t0 = time.time()
    obj_func(init_pos.flatten())
    t1 = time.time()
    time_per_eval = t1 - t0
    
    # Population size 10 (smaller for speed)
    popsize = 10
    n_evals_per_iter = popsize * 128
    
    # Assume 5s per iteration (conservative)
    # We have 600s.
    # maxiter = 600 / 5 = 120?
    # Let's set maxiter high and use a callback to stop if time runs out.
    
    stop_flag = [False]
    
    def callback(xk, convergence):
        if time.time() - start_time > deadline - 10:
            stop_flag[0] = True
            return True
        return False
        
    # DE can be slow. 
    # Alternative: Use Nelder-Mead or L-BFGS-B starting from the DE result?
    # Or just use DE with a reasonable maxiter.
    
    # Let's use a smaller population to allow more iterations.
    # popsize=5?
    
    try:
        # We'll run DE
        # tol=1e-5, maxiter=100
        # We'll wrap it to respect the deadline.
        
        # Since DE can hang, we run it in a way that we can interrupt?
        # No, we just set maxiter based on estimated time.
        
        # Let's estimate:
        # 10000 samples, 64 tubes.
        # Vectorized ops are fast.
        # Let's assume 1ms per eval.
        # Popsize 10 -> 1280 evals/iter -> 1.3s/iter.
        # 600s / 1.3s = ~400 iterations.
        
        result = differential_evolution(
            obj_func, 
            bounds, 
            maxiter=300, 
            popsize=10, 
            tol=1e-6,
            seed=42,
            callback=callback,
            polish=True  # Use L-BFGS-B to polish the result
        )
        
        if result.success or (time.time() - start_time < deadline - 5):
            best_pos = result.x.reshape(64, 2)
        else:
            best_pos = init_pos
            
    except Exception as e:
        print(f"DE failed: {e}")
        best_pos = init_pos
        
    # Final Polish with Nelder-Mead or L-BFGS-B if time permits
    if time.time() - start_time < deadline - 20:
        try:
            from scipy.optimize import minimize
            
            # Use L-BFGS-B
            res = minimize(
                obj_func, 
                best_pos.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-5}
            )
            
            if res.fun < obj_func(best_pos.flatten()):
                best_pos = res.x.reshape(64, 2)
                
        except:
            pass
            
    # Ensure finite
    if not np.all(np.isfinite(best_pos)):
        best_pos = init_pos
        
    return best_pos

if __name__ == "__main__":
    # Quick test
    res = kakeya_needle_3d8()
    print(res.shape)
    print(f"Volume: {fast_volume_estimate(res)}")
