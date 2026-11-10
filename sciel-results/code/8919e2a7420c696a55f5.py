# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _compute_area(x, a, y, w, L_base):
    """
    Compute the area of the union of triangles.
    x: (64,) array of base positions
    a: (64,) array of slopes (a_i)
    y: (20000,) array of heights
    w: (20000,) array of widths at height y
    L_base: (64, 20000) array where L_base[i, m] = a[i] * y[m]
    """
    # L[i, m] = x[i] + a[i] * y[m]
    L = L_base + x[:, np.newaxis]
    
    # Sort L along axis 0 (for each height m, sort the 64 left edges)
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted edges
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length at each height:
    # union_len[m] = w[m] + sum_i min(d[i, m], w[m])
    mins = np.minimum(d, w[np.newaxis, :])
    sum_mins = np.sum(mins, axis=0)
    union_lens = w + sum_mins
    
    # Area is mean of union lengths
    return np.mean(union_lens)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns 64 finite floats x_0..x_63.
    """
    n = 64
    k = int(np.log2(n))
    
    # Precompute static parts
    a = np.arange(1, n + 1, dtype=np.float64) / n
    num_heights = 20000
    y = (np.arange(num_heights, dtype=np.float64) + 0.5) / num_heights
    w = (1.0 - y) / n
    
    # L_base[i, m] = a[i] * y[m]
    L_base = np.outer(a, y)
    
    # Initial guess: Keich construction (used as a starting point or seed)
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s margin before 600s kill
    
    best_x = x_keich.copy()
    best_area = _compute_area(x_keich, a, y, w, L_base)
    
    # Objective function for Differential Evolution
    def objective(x):
        # Check for deadline
        if time.time() > deadline:
            return best_area
        return _compute_area(x, a, y, w, L_base)
    
    # Define bounds for x_i
    # The triangles are [x_i, x_i+1/64] on base. 
    # To keep them somewhat within reasonable range, say [-1, 2]
    bounds = [(-1.0, 2.0)] * n
    
    # Run Differential Evolution
    # DE is a global optimization method, good for non-convex problems.
    # maxiter controls the number of generations.
    # popsize controls the population size.
    # We want to use as much time as possible.
    # Let's estimate time per function eval.
    # _compute_area involves sorting 64x20000 array.
    # Sorting 20000 vectors of size 64 is roughly 20000 * 64 * log(64) ~ 20000 * 64 * 6 = 7.68M ops.
    # Plus overhead. Let's say ~1ms per eval? 
    # If 1ms, we can do ~1000 evals/sec. 
    # 590 sec * 1000 = 590,000 evals.
    # DE with popsize=20 (1280 params), maxiter=100 -> 128,000 evals.
    # This is feasible.
    
    try:
        # Use a callback to monitor time and potentially stop early if no improvement?
        # DE has a 'maxiter' and 'popsize'.
        # Let's set a high maxiter and rely on time limit if needed, 
        # but DE doesn't have a built-in time limit. 
        # We can use a callback to raise an exception if time runs out, 
        # but that might be messy. 
        # Instead, let's estimate a safe maxiter.
        
        # Let's do a quick benchmark to be safe? No, keep it simple.
        # popsize=30, maxiter=200 is a reasonable start.
        # If it's too slow, we might timeout.
        # Let's use popsize=20, maxiter=100 first.
        
        res = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            init='sobol', # Use Sobol sequence for better initial spread
            # polish=True, # Default is True, uses L-BFGS-B at the end. 
                          # This is good: DE finds global basin, L-BFGS-B polishes.
            # But L-BFGS-B might be slow if it gets stuck in local minima of the non-smooth function?
            # The function is piecewise linear (sort of), so gradients are undefined at boundaries.
            # L-BFGS-B might struggle with the non-smoothness.
            # Let's disable polish and do our own polish? Or trust it?
            # The previous program used L-BFGS-B on a smooth surrogate.
            # Here DE explores the raw function.
            # Let's try polish=False first to avoid getting stuck in a bad local min via L-BFGS-B,
            # and then do a manual coordinate descent or a second DE run with smaller bounds?
            # Actually, the raw function is continuous but not differentiable.
            # L-BFGS-B assumes differentiability. It might work okay or might fail.
            # Let's set polish=False to be safe, and then do a simple coordinate descent.
            polish=False
        )
        
        if time.time() < deadline:
            x_de = res.x
            area_de = objective(x_de)
            if np.all(np.isfinite(x_de)) and area_de < best_area:
                best_x = x_de.copy()
                best_area = area_de
                
        # Phase 2: Polish with Coordinate Descent on the exact function
        # This handles the non-smoothness better than L-BFGS-B might.
        if time.time() < deadline:
            x_cd = best_x.copy()
            improved = True
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
            
            while improved and time.time() < deadline - 10:
                improved = False
                for size in step_sizes:
                    for i in range(n):
                        if time.time() > deadline - 10:
                            break
                        # Try +/- step
                        for delta in [size, -size]:
                            x_try = x_cd.copy()
                            x_try[i] += delta
                            if x_try[i] < bounds[i][0] or x_try[i] > bounds[i][1]:
                                continue
                            area_try = objective(x_try)
                            if area_try < best_area - 1e-15:
                                x_cd[i] = x_try[i]
                                best_area = area_try
                                improved = True
                                break # Move to next i or smaller step? 
                                     # Usually restart the loop for this size?
                                     # Or just continue scanning?
                                     # Let's break out of delta loop, but continue i loop?
                                     # Actually, if we improve, the landscape changes.
                                     # Let's restart the size loop? No, that's expensive.
                                     # Let's just mark improved and continue.
            
            if np.all(np.isfinite(x_cd)) and best_area < _compute_area(best_x, a, y, w, L_base):
                best_x = x_cd.copy()
                
    except Exception as e:
        pass
    
    # Final verification
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
