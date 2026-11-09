import numpy as np
import time
from scipy.optimize import minimize

def _union_area_correct(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    Uses sorting to handle the union correctly.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    # L[i, m] = x[i] + a[i] * y[m]
    # Shape: (n, M)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0 (the triangle index axis) for each height m
    # L_sorted[k, m] is the k-th smallest L value at height m
    L_sorted = np.sort(L, axis=0)
    
    # Gaps between consecutive sorted triangles
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length at each height:
    # The union of intervals [L_i, L_i + w] is w + sum(min(gap, w)) for consecutive sorted intervals
    # Note: The first interval contributes w. Each subsequent interval contributes min(gap, w).
    # So total length = w + sum_{k=1}^{n-1} min(d[k], w)
    min_d = np.minimum(d, w[np.newaxis, :])
    union_len = w[np.newaxis, :] + np.sum(min_d, axis=0)
    
    area = np.mean(union_len)
    return float(area)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill
    
    # Precompute constants for the area function to avoid re-computation overhead if possible,
    # but _union_area_correct is self-contained. We can pass them to a closure for speed.
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n

    def area_obj(x):
        # x is a flat array of length n
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        min_d = np.minimum(d, w[np.newaxis, :])
        union_len = w[np.newaxis, :] + np.sum(min_d, axis=0)
        return float(np.mean(union_len))

    best_x = None
    best_area = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Try several starting points
    # 1. Zeros
    # 2. Random small values
    # 3. Values spread out to reduce initial overlap
    
    starts = [
        np.zeros(n),
        rng.normal(0, 0.1, n),
        rng.normal(0, 0.5, n),
        np.linspace(-0.5, 0.5, n), # Spread out
        np.linspace(0, 1, n) - 0.5
    ]
    
    # Add some random restarts
    for _ in range(5):
        starts.append(rng.normal(0, 1.0, n))
        
    for x0 in starts:
        if time.time() > deadline - 30:
            break
            
        try:
            # Use Powell method, which is robust for non-smooth functions
            # maxiter is limited by time
            result = minimize(area_obj, x0, method='Powell', options={'maxiter': 1000, 'disp': False})
            
            if result.success or result.fun < best_area:
                if np.all(np.isfinite(result.x)):
                    current_area = result.fun
                    if current_area < best_area:
                        best_area = current_area
                        best_x = result.x.copy()
        except Exception:
            continue
            
    # If optimization didn't improve, try a simple local search from the best found
    if best_x is None:
        best_x = np.zeros(n)
        best_area = area_obj(best_x)
        
    # Perform a final local search (coordinate descent) to polish
    # For each i, try to adjust x_i to minimize area
    # Since the function is piecewise linear, the minimum is at a kink or where gradient is 0.
    # A simple grid search around the current best for each coordinate might be effective.
    
    # Let's do a few iterations of coordinate descent with a small step
    step_size = 0.01
    for _ in range(5): # 5 passes
        if time.time() > deadline - 10:
            break
        for i in range(n):
            if time.time() > deadline - 10:
                break
            # Try moving x[i] in small steps
            best_local = best_x[i]
            best_local_area = best_area
            # Check neighborhood
            for delta in [-step_size, -step_size*0.5, step_size*0.5, step_size]:
                trial_x = best_x.copy()
                trial_x[i] = best_x[i] + delta
                trial_area = area_obj(trial_x)
                if trial_area < best_local_area:
                    best_local_area = trial_area
                    best_local = trial_x[i]
            best_x[i] = best_local
            best_area = best_local_area
            
    # Final validation
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n)
        
    final_area = area_obj(best_x)
    if not np.isfinite(final_area) or final_area <= 0:
        return np.zeros(n)
        
    return best_x.astype(np.float64)
