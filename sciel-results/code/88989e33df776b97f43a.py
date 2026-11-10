import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Uses the Keich construction as an initial guess and refines it using 
    scipy.optimize.differential_evolution (global) followed by L-BFGS-B (local).
    """
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate initial guess using Keich construction
    def keich_initialization(n):
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)

    x0 = keich_initialization(n)

    # 2. Define the area calculation function
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    w_col = w[:, None]

    def compute_area(x):
        # x: shape (n,)
        # L: shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        # Compute gaps between consecutive sorted L
        gaps = np.diff(L_sorted, axis=1)
        # Union length = w + sum(min(gap, w))
        min_gaps = np.minimum(gaps, w_col)
        union_len = w + np.sum(min_gaps, axis=1)
        # Area is mean of union lengths
        return np.mean(union_len)

    best_x = x0
    best_val = compute_area(best_x)
    
    start_time = time.time()
    timeout = 580  # Safety margin for 600s limit

    # 3. Global Optimization with Differential Evolution
    # DE is robust to non-smooth functions (sorting)
    # Bounds: Keich values are small, but allow some range
    bounds = [(-1.0, 1.0)] * n
    
    try:
        # Use a moderate population size and iterations to fit within budget
        # popsize=10 means 10*n = 640 individuals
        # maxiter=50 means 50 generations
        # Total evals ~ 32000
        # This should be fast enough to leave time for local optimization
        de_res = differential_evolution(
            compute_area, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=0.01, 
            seed=42,
            init='latinhypercube'
        )
        
        if de_res.fun < best_val:
            best_x = de_res.x
            best_val = de_res.fun
            
    except Exception as e:
        pass

    # 4. Local Optimization with L-BFGS-B
    # Polish the result from DE (or Keich if DE failed)
    if time.time() - start_time < timeout - 50:
        try:
            res2 = minimize(
                compute_area, 
                best_x, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'disp': False, 'maxfun': 2000}
            )
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception as e:
            pass

    # 5. Final Local Optimization with Nelder-Mead if time permits
    # Nelder-Mead can sometimes find better local minima than gradient methods
    # for non-smooth functions
    if time.time() - start_time < timeout - 20:
        try:
            res3 = minimize(
                compute_area, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res3.fun < best_val:
                best_x = res3.x
                best_val = res3.fun
        except Exception as e:
            pass

    # Final check for validity
    if not np.all(np.isfinite(best_x)):
        return x0 # Fallback to baseline
        
    return best_x

# EVOLVE-BLOCK-END
