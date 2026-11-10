# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_area(x):
    """
    Compute the area of the union of 64 triangles.
    x: shape (64,)
    Returns: scalar area
    """
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    # L_i(y) = x_i + a_i * y
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n
    
    # Sort L along axis 0 for each y
    L_sorted = np.sort(L, axis=0)
    
    # Gaps between consecutive sorted L's
    gaps = np.diff(L_sorted, axis=0)  # Shape: (63, 20000)
    
    # Union length contribution
    # Union length = w + sum(min(gaps, w))
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    
    area = union_len.mean()
    return area

def _keich_init():
    """
    Generate the initial Keich construction positions.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Initial guess from Keich construction
    x0 = _keich_init()
    
    best_x = x0.copy()
    best_area = _calc_area(best_x)
    
    # We will use Powell's method, which is derivative-free and handles non-smoothness well.
    # We run multiple restarts to avoid local minima.
    
    # Strategy: 
    # 1. Run Powell from Keich init.
    # 2. Run Powell from a few random perturbations of the Keich init.
    # 3. Keep the best result.
    
    max_restarts = 5
    time_per_restart = 80  # seconds
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution found so far, or the initial one
            # Add small noise to escape potential local minima
            noise_scale = 0.05 * (restart + 1)
            x_start = best_x + np.random.normal(0, noise_scale, size=64)
            
        # Run Powell's method
        # Powell is robust to non-differentiability
        # maxiter is a safety limit, but we also check time
        # ftol and xtol are tolerances for convergence
        
        try:
            res = minimize(
                _calc_area, 
                x_start, 
                method='Powell', 
                options={
                    'maxiter': 2000, 
                    'ftol': 1e-12, 
                    'xtol': 1e-12,
                    'disp': False
                }
            )
            
            if res.success or res.fun < best_area:
                if res.fun < best_area:
                    best_area = res.fun
                    best_x = res.x.copy()
                    
        except Exception as e:
            # If optimization fails, continue to next restart
            continue
            
        # Check time
        if time.time() > deadline - 20:
            break
            
    # Final validation
    if not np.all(np.isfinite(best_x)):
        return x0
        
    return best_x

# EVOLVE-BLOCK-END
