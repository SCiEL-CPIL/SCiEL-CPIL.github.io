# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
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
    deadline = start_time + 620.0  # Leave 10s margin before 630s kill
    
    best_x = x_keich.copy()
    best_area = _compute_area(x_keich, a, y, w, L_base)
    
    # Define the objective function for the optimizer
    def objective(x):
        return _compute_area(x, a, y, w, L_base)
    
    # Bounds for x_i. The triangles can be shifted significantly.
    # Based on geometry, x_i in [-1, 2] is a reasonable search space.
    bounds = [(-1.5, 2.5)] * n
    
    try:
        # Strategy: Differential Evolution
        # DE is a population-based global optimizer. It is slower per step but
        # much better at escaping local minima than L-BFGS-B.
        # We use 'best1bin' strategy with moderate mutation/crossover.
        
        # To manage time, we limit maxiter. 
        # DE with popsize=15 (default) * 64 vars * maxiter evaluations.
        # Each eval is ~0.01s (sort 64x20000 is fast in numpy).
        # 15 * 64 * 100 iters = 96,000 evals * 0.01s = 960s. Too slow.
        # We need to be very careful with maxiter.
        # Let's aim for ~3000-5000 evaluations total.
        # popsize=10, maxiter=50 -> 10*50 = 500 generations? No, popsize is number of individuals.
        # DE performs popsize evaluations per generation.
        # If popsize=10, maxiter=20 -> 200 evaluations. Very fast.
        # Let's try popsize=20, maxiter=50 -> 1000 evals. ~10-20s.
        # We can afford more.
        
        # Let's run DE with a timeout check inside the callback if possible, 
        # or just set a conservative maxiter.
        
        # Callback to check time
        def callback(xk, convergence=True):
            if time.time() > deadline - 10:
                return True # Stop early
            return False

        res_de = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            atol=1e-12,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            init='latinhypercube',
            callback=callback,
            polish=False # We will do polish separately to control time
        )
        
        if time.time() < deadline - 5:
            x_de = res_de.x
            area_de = objective(x_de)
            if np.all(np.isfinite(x_de)) and area_de < best_area:
                best_x = x_de.copy()
                best_area = area_de
        
        # Phase 2: Local Polish with Nelder-Mead
        # Nelder-Mead is derivative-free and works well for non-smooth functions
        # if the step size is managed well. It is faster than DE per iteration
        # but local.
        
        if time.time() < deadline - 10:
            # Use the best result from DE or Keich
            x_start = best_x.copy()
            
            # Nelder-Mead options
            res_nm = minimize(
                objective,
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if time.time() < deadline:
                x_nm = res_nm.x
                area_nm = objective(x_nm)
                if np.all(np.isfinite(x_nm)) and area_nm < best_area:
                    best_x = x_nm.copy()
                    best_area = area_nm

        # Phase 3: Coordinate Descent Polish (if time remains)
        # This can squeeze out a bit more by fine-tuning individual coordinates.
        if time.time() < deadline - 2:
            x_cd = best_x.copy()
            improved = True
            max_cd_iters = 10
            for cd_iter in range(max_cd_iters):
                if time.time() > deadline - 2:
                    break
                any_improve = False
                # Shuffle order to avoid bias
                indices = np.random.permutation(n)
                for i in indices:
                    if time.time() > deadline - 2:
                        break
                    # Try a few small steps
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                        x_try = x_cd.copy()
                        x_try[i] += delta
                        area_try = objective(x_try)
                        if area_try < best_area - 1e-14:
                            x_cd[i] = x_try[i]
                            best_area = area_try
                            any_improve = True
                            break
                if not any_improve:
                    break
            
            if np.all(np.isfinite(x_cd)) and best_area < _compute_area(best_x, a, y, w, L_base):
                best_x = x_cd.copy()
                
    except Exception as e:
        # If anything fails, fall back to best known
        pass
    
    # Final verification
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
