import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    For position index i (0-based) let a_i = (i + 1) / 64 and define the triangle R_i
    with vertices (x_i, 0), (x_i + 1/64, 0), (x_i + a_i, 1).
    
    The objective is to choose x = (x_0, ..., x_63) so that the AREA of the union of
    the 64 triangles is as small as possible.
    
    This implementation:
    1. Initializes with the Keich construction (seed).
    2. Defines a fast approximation of the union area using quadrature.
    3. Uses scipy.optimize.minimize (L-BFGS-B) to refine the positions.
    4. Performs multiple restarts with random perturbations to escape local minima.
    5. Returns the best position array found within the time budget.
    """
    n = 64
    k = int(np.log2(n))  # k = 6
    
    # Generate seed positions (Keich construction)
    j = np.arange(1, k + 1)
    seed_positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        seed_positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    a = np.arange(1, n + 1) / n  # a_i = (i+1)/n

    def union_area(x, num_heights=200):
        """
        Compute the union area of the 64 triangles using quadrature at num_heights points.
        x: array of shape (n,)
        """
        ys = (np.arange(num_heights) + 0.5) / num_heights
        
        total_area = 0.0
        for y in ys:
            w = (1.0 - y) / n
            Ls = x + a * y
            sorted_Ls = np.sort(Ls)
            diffs = np.diff(sorted_Ls)
            extensions = np.minimum(w, diffs)
            union_len = w + np.sum(extensions)
            total_area += union_len
            
        return total_area / num_heights

    def objective(x):
        return union_area(x)

    start_time = time.time()
    best_x = seed_positions.copy()
    best_area = objective(best_x)
    
    # Optimization settings
    options = {'maxiter': 1000, 'disp': False}
    
    max_time = 620.0  # Safety margin before the 630s limit
    num_restarts = 50  # We will break if time runs out
    
    # Pre-compute some useful arrays
    np.random.seed(42)  # For reproducibility
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time - 10:
            break
            
        if restart == 0:
            x0 = seed_positions.copy()
        elif restart < 10:
            # Small perturbations of seed
            sigma = 0.05 / (1 + restart * 0.1)
            x0 = seed_positions + np.random.normal(0, sigma, n)
        elif restart < 20:
            # Medium perturbations
            x0 = seed_positions + np.random.normal(0, 0.2, n)
        else:
            # Random starts in a reasonable range
            x0 = np.random.uniform(-0.5, 1.5, n)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options=options)
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with higher resolution if time allows
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 5:
        def union_area_high_res(x, num_heights=2000):
            ys = (np.arange(num_heights) + 0.5) / num_heights
            total_area = 0.0
            for y in ys:
                w = (1.0 - y) / n
                Ls = x + a * y
                sorted_Ls = np.sort(Ls)
                diffs = np.diff(sorted_Ls)
                extensions = np.minimum(w, diffs)
                union_len = w + np.sum(extensions)
                total_area += union_len
            return total_area / num_heights
        
        def objective_high_res(x):
            return union_area_high_res(x)
        
        try:
            res = minimize(objective_high_res, best_x, method='L-BFGS-B', 
                          options={'maxiter': 500, 'disp': False})
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x, dtype=np.float64)
