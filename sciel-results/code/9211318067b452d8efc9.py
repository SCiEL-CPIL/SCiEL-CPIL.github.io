import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _mc_volume(params, n_samples=50000, seed=12345):
    """
    Fast Monte Carlo volume estimate for a given set of base positions.
    params: array of shape (16,) -> f[0:8], g[8:16]
    """
    f = params[:8]
    g = params[8:16]
    N = 8
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    
    # Construct base positions (64, 2)
    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x = f[i_flat]
    y = g[j_flat]
    
    # Determine bounding box
    # X range for tube k (i, j): [x_k + c*i/8, x_k + c*i/8 + 1/8]
    # At c=0: [x_k, x_k + 1/8]
    # At c=1: [x_k + i/8, x_k + i/8 + 1/8]
    # Min x for tube k is min(x_k, x_k + i/8) = x_k + min(0, i/8) = x_k (since i>=0)
    # Max x for tube k is max(x_k + 1/8, x_k + i/8 + 1/8) = x_k + max(1/8, i/8 + 1/8) = x_k + (i+1)/8
    
    x_min = np.min(x)
    x_max = np.max(x + (i_flat + 1) / N)
    y_min = np.min(y)
    y_max = np.max(y + (j_flat + 1) / N)
    
    if x_max - x_min <= 1e-9 or y_max - y_min <= 1e-9:
        return 1e9
        
    rng = np.random.default_rng(seed)
    a = rng.uniform(x_min, x_max, n_samples)
    b = rng.uniform(y_min, y_max, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside at least one tube
    # Tube (i, j) at height c is:
    # [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8]
    
    # We can vectorize this check.
    # For each sample (a, b, c), check all 64 tubes.
    # This might be slow if done naively (50000 * 64 * 2 comparisons).
    # Let's try to optimize.
    
    # Alternative: Loop over samples? No, vectorize over tubes for each sample?
    # Or loop over tubes and mark samples?
    
    # Let's try a loop over tubes, accumulating a mask.
    # in_any = np.zeros(n_samples, dtype=bool)
    # for k in range(64):
    #    i = k // 8
    #    j = k % 8
    #    x_k = x[k]
    #    y_k = y[k]
    #    x_start = x_k + c * i / N
    #    x_end = x_start + 1.0 / N
    #    y_start = y_k + c * j / N
    #    y_end = y_start + 1.0 / N
    #    in_tube = (a >= x_start) & (a <= x_end) & (b >= y_start) & (b <= y_end)
    #    in_any |= in_tube
    #    if in_any.all(): break
    
    # This is 64 * (4 comparisons + 3 logical ops) * 50000 = ~10M ops. Fast enough in NumPy.
    
    in_any = np.zeros(n_samples, dtype=bool)
    for k in range(64):
        i = k // 8
        j = k % 8
        x_start = x[k] + c * (i / N)
        x_end = x_start + 1.0 / N
        y_start = y[k] + c * (j / N)
        y_end = y_start + 1.0 / N
        
        in_tube = (a >= x_start) & (a <= x_end) & (b >= y_start) & (b <= y_end)
        in_any |= in_tube
        # Early termination is hard with vectorized boolean array unless we check sum
        # if np.count_nonzero(in_any) == n_samples: break
        
    fraction = np.count_nonzero(in_any) / n_samples
    V = fraction * (x_max - x_min) * (y_max - y_min)
    
    return V

def kakeya_needle_3d8() -> np.ndarray:
    deadline = time.time() + 580
    
    # Strategy:
    # 1. Use Differential Evolution on the 16 parameters (f, g) with a fast MC objective.
    # 2. Polish the result with L-BFGS-B using a gradient-free approximation or just tighter DE.
    #    Actually, DE doesn't provide gradients. We can use Nelder-Mead or Powell for polishing.
    
    bounds = [(-1.0, 1.0)] * 16
    
    # Initial population
    # Heuristic: f(i) = -i/16, g(j) = -j/16
    init_params = np.zeros(16)
    i_vals = np.arange(8)
    init_params[:8] = -i_vals / 16.0
    init_params[8:] = -i_vals / 16.0
    
    # DE with MC objective
    # n_samples=50000 is a bit slow for DE (popsize=20, maxiter=100 -> 2000 evals * 0.1s = 200s).
    # Let's use fewer samples for DE, then polish with more samples.
    
    def objective_de(params):
        return _mc_volume(params, n_samples=10000, seed=12345)
        
    def objective_polish(params):
        return _mc_volume(params, n_samples=50000, seed=12345)

    try:
        result = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6,
            polish=False,
            init='sobol'
        )
        best_params = result.x
    except Exception:
        best_params = init_params
        
    # Check time
    if time.time() > deadline - 60:
        # Not enough time for polishing, return DE result
        pass
    else:
        # Polish with Nelder-Mead or Powell
        try:
            res = minimize(
                objective_polish, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < objective_polish(best_params):
                best_params = res.x
        except Exception:
            pass
            
    f = best_params[:8]
    g = best_params[8:16]
    
    # Construct positions
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x = f[i_flat]
    y = g[j_flat]
    
    positions = np.stack([x, y], axis=1)
    
    return positions
