# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_length_1d(positions, c, N=8):
    """
    Compute the length of the union of N intervals at height c.
    positions: array of shape (N,) of base positions x_i.
    Interval i is [x_i + c*i/N, x_i + 1/N + c*i/N].
    """
    # starts and ends
    # s_i = x_i + c * i / N
    # e_i = s_i + 1/N
    idx = np.arange(N)
    starts = positions + c * idx / N
    ends = starts + 1.0 / N
    
    # Sort by start
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = ends[order]
    
    # Merge intervals
    # Initialize with first
    current_start = s_sorted[0]
    current_end = e_sorted[0]
    total_len = 0.0
    
    for k in range(1, N):
        if s_sorted[k] <= current_end:
            # Overlap or touch
            if e_sorted[k] > current_end:
                current_end = e_sorted[k]
        else:
            # Disjoint
            total_len += current_end - current_start
            current_start = s_sorted[k]
            current_end = e_sorted[k]
            
    total_len += current_end - current_start
    return total_len

def _compute_volume(x_base, y_base, N=8, num_samples=100):
    """
    Compute the volume of the union of tubes.
    V = integral_0^1 L_x(c) L_y(c) dc
    """
    # Use Gauss-Legendre quadrature for better accuracy with fewer points
    # or just simple trapezoidal/Simpson.
    # N=8 is small, so even 50 points is enough for high precision.
    c_vals, w_vals = np.polynomial.legendre.leggauss(num_samples)
    # Map c_vals from [-1, 1] to [0, 1]
    c_mapped = 0.5 * (c_vals + 1.0)
    w_mapped = 0.5 * w_vals
    
    vol = 0.0
    for c, w in zip(c_mapped, w_mapped):
        lx = _union_length_1d(x_base, c, N)
        ly = _union_length_1d(y_base, c, N)
        vol += w * lx * ly
        
    return vol

def _objective(params):
    """
    Objective function for the optimizer.
    params: array of shape (16,) = [x_0..x_7, y_0..y_7]
    """
    x_base = params[:8]
    y_base = params[8:]
    return _compute_volume(x_base, y_base)

def kakeya_needle_3d8() -> np.ndarray:
    """
    Returns base positions for 64 tubes.
    """
    N = 8
    start_time = time.time()
    time_limit = 590.0  # Safety margin
    
    # Initial guess: The seed solution (Keich)
    # But we can also try random or simple linear.
    # Let's use the seed solution as the primary start.
    # Reproduce Keich
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # The seed solution is x_i = keich[i], y_j = keich[j]
    # This corresponds to params = [keich, keich]
    initial_params = np.concatenate([keich, keich])
    
    # We can also try a few other starting points to escape local minima
    # E.g. uniform spacing, or random
    # But the seed is likely close to the global minimum or a good local one.
    
    # Optimization
    # Nelder-Mead is derivative-free and good for low dimensions.
    # Powell is also good.
    # Let's use Nelder-Mead with adaptive settings.
    
    best_params = initial_params.copy()
    best_val = _objective(best_params)
    
    # Multi-start with a few perturbations
    rng = np.random.default_rng(42)
    
    # List of starting points
    starts = [initial_params]
    
    # Add some random perturbations
    for _ in range(5):
        perturbed = initial_params + rng.normal(0, 0.1, size=16)
        starts.append(perturbed)
        
    # Add a "uniform" start: x_i = i * small_step? 
    # Actually, just random is fine.
    
    for start_p in starts:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(_objective, start_p, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Check time again
    if time.time() - start_time > time_limit:
         # Return best found
         pass

    # Construct the 64x2 array
    x_base = best_params[:8]
    y_base = best_params[8:]
    
    # i = k // 8, j = k % 8
    # Row k is tube (i, j)
    # x_k = x_base[i], y_k = y_base[j]
    
    i_idx = np.arange(64) // 8
    j_idx = np.arange(64) % 8
    
    base_x = x_base[i_idx]
    base_y = y_base[j_idx]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
