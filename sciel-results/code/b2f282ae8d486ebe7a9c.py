import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Combines the Keich construction and Aligned initialization with a robust
    derivative-free local search (Nelder-Mead) to find a better local minimum.
    """
    n = 64
    k = int(np.log2(n))
    
    # Pre-compute constants for area calculation
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n

    def compute_area(x):
        """
        Computes the area of the union of the triangles.
        x: shape (n,)
        """
        # L_i(y) = x_i + a_i * y
        # L: shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        
        # Compute gaps between consecutive sorted L
        # Gaps shape (M, n-1)
        gaps = np.diff(L_sorted, axis=1)
        
        # Union length = w + sum(min(gap, w))
        # w shape (M, 1)
        w_col = w[:, None]
        min_gaps = np.minimum(gaps, w_col)
        union_len = w + np.sum(min_gaps, axis=1)
        
        # Area is mean of union lengths
        return np.mean(union_len)

    # 1. Generate initial guesses
    inits = []
    
    # Keich Initialization
    def keich_initialization(n):
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)
    
    inits.append(keich_initialization(n))
    
    # Aligned Initialization: x_i = 0.5 - a_i
    x_aligned = 0.5 - a
    inits.append(x_aligned)
    
    # Another potential start: x_i = -a_i (converging to 0)
    x_conv0 = -a
    inits.append(x_conv0)
    
    # Random initialization
    np.random.seed(42)
    x_rand = np.random.uniform(-0.5, 0.5, n)
    inits.append(x_rand)

    # 2. Optimize each initialization
    best_x = inits[0]
    best_val = compute_area(best_x)
    
    start_time = time.time()
    timeout = 600  # Safety margin
    
    # Use Nelder-Mead for derivative-free optimization
    # This is more robust for the non-smooth union area function
    nm_options = {
        'maxiter': 500,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    for x0 in inits:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = minimize(
                compute_area, 
                x0, 
                method='Nelder-Mead', 
                options=nm_options
            )
            
            if res.success and res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    # If time permits, try a second pass with L-BFGS-B on the best found so far
    # L-BFGS-B might refine the solution if we are far from the non-smooth boundaries
    if time.time() - start_time < timeout - 50:
        try:
            res2 = minimize(
                compute_area, 
                best_x, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'disp': False, 'maxfun': 1000}
            )
            if res2.success and res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Final check for validity
    if not np.all(np.isfinite(best_x)):
        return inits[0] # Fallback to Keich
        
    return best_x
