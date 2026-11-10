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
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Optimization
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s margin before 600s kill
    
    best_x = x_keich.copy()
    best_area = _compute_area(x_keich, a, y, w, L_base)
    
    # Define the objective function for scipy
    def objective(x):
        return _compute_area(x, a, y, w, L_base)
    
    # Bounds for differential evolution
    # The triangles must fit reasonably. 
    # x_i is the left base. Right base is x_i + 1/64. Apex is x_i + a_i.
    # To keep them in [0, 1] roughly, x_i should be in [-1, 1] or similar.
    # Let's use a slightly wider range to allow exploration.
    bounds = [(-1.0, 2.0)] * n
    
    try:
        # Use Differential Evolution for global search
        # maxiter is set to allow enough iterations within the time budget.
        # Each iteration involves `popsize` evaluations.
        # popsize=15 (default is 15*n, which is too large). Let's use 10-20.
        # With n=64, popsize=15 means ~960 evaluations per generation.
        # 200 generations * 960 = 192,000 evaluations.
        # Each evaluation is O(64*20000 log 64) ~ 10^7 ops.
        # 192,000 * 10^7 = 1.9 * 10^12 ops. This might be too slow in Python/NumPy if not vectorized well, 
        # but _compute_area is vectorized. 
        # Let's estimate: 1 eval ~ 1-2 ms. 192,000 evals ~ 200-400 seconds.
        # We need to be careful with maxiter and popsize.
        
        # Let's try a smaller popsize and fewer iterations first, then polish.
        # popsize=10, maxiter=100.
        # 10 * 100 * 64 (no, popsize is number of individuals, not dimension).
        # popsize=10 means 10 individuals.
        # 100 iterations * 10 individuals = 1000 evaluations.
        # This is very fast. We can afford more.
        
        # Let's use popsize=20, maxiter=200.
        # 20 * 200 = 4000 evaluations. ~4-8 seconds.
        # We have 590s. We can do much more.
        
        # Let's aim for ~3000 evaluations total for DE.
        # popsize=25, maxiter=120 -> 3000 evals.
        
        res = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,  # Use L-BFGS-B to polish the best result
            init='latinhypercube',
            updating='immediate',
            callback=None,
            workers=1
        )
        
        if time.time() < deadline:
            x_de = res.x
            area_de = objective(x_de)
            if np.all(np.isfinite(x_de)) and area_de < best_area:
                best_x = x_de.copy()
                best_area = area_de

        # If we have time, do a second round of DE with tighter bounds or different params?
        # Or just polish further?
        # The 'polish=True' in DE already does L-BFGS-B.
        
        # Let's try a local polish from the best DE result using L-BFGS-B again, 
        # just in case DE's internal polish was cut short or didn't converge fully.
        
        if time.time() < deadline - 60:
            from scipy.optimize import minimize
            
            res2 = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 50,
                    'maxcor': 50
                }
            )
            
            x2 = res2.x
            area2 = objective(x2)
            if np.all(np.isfinite(x2)) and area2 < best_area:
                best_x = x2.copy()
                best_area = area2

    except Exception as e:
        pass
    
    # Final verification
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
