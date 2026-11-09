# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_area(x, n=64, M=20000):
    """
    Compute the area of the union of the n triangles defined by base positions x.
    Uses the quadrature method specified in the task.
    """
    # y_m = (m + 0.5) / M for m in 0..M-1
    m = np.arange(M)
    y = (m + 0.5) / M  # shape (M,)
    
    # a_i = (i + 1) / n for i in 0..n-1
    a = (np.arange(n) + 1) / n  # shape (n,)
    
    # L_i(y) = x_i + a_i * y for each i and each y
    # We need to compute L for all i and all y, then sort along the i-axis for each y.
    # L has shape (M, n)
    L = x[None, :] + a[None, :] * y[:, None]  # broadcasting: (1,n) + (M,1)*(1,n) -> (M,n)
    
    w = (1 - y) / n  # shape (M,)
    
    # Sort L along axis 1 for each y
    L_sorted = np.sort(L, axis=1)  # shape (M, n)
    
    # Compute gaps between consecutive sorted L values
    gaps = L_sorted[:, 1:] - L_sorted[:, :-1]  # shape (M, n-1)
    
    # union length = w + sum over j of min(gaps_j, w)
    # min(gaps, w) where w is broadcast
    min_gaps = np.minimum(gaps, w[:, None])  # shape (M, n-1)
    
    union_lengths = w + np.sum(min_gaps, axis=1)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Start with the Keich construction as initial guess
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    best_x = positions.copy()
    best_area = _compute_area(best_x, n, 20000)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds
    
    # Gradient-free optimization using Nelder-Mead with a smooth surrogate
    # We use scipy's minimize with Nelder-Mead or BFGS if we can compute gradients
    # Since the objective is non-smooth (due to min and sort), we use Nelder-Mead
    # but with a smooth surrogate if possible.
    
    # Let's try scipy.optimize.minimize with Nelder-Mead
    try:
        from scipy.optimize import minimize
        import scipy
    except ImportError:
        # Fallback to a simple local search
        # ...
        return best_x
    
    # Define the objective function
    def objective(x):
        return _compute_area(x, n, 20000)
    
    # Use Nelder-Mead with adaptive settings
    # Since the function is non-smooth, Nelder-Mead is appropriate
    # We'll run multiple restarts with different initial points if time permits
    
    best_area = _compute_area(best_x, n, 20000)
    best_x = best_x.copy()
    
    # Try a few different optimization strategies
    # 1. Nelder-Mead from the Keich starting point
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        try:
            result = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # 2. Try L-BFGS-B with finite differences (may work despite non-smoothness)
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        try:
            result = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # 3. Try a local random perturbation search (simulated annealing style)
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        x = best_x.copy()
        current_area = best_area
        T = 1e-4  # initial temperature
        cooling = 0.999
        step_size = 0.01
        
        for step in range(10000):
            if time.time() > deadline - safety_margin:
                break
            
            # Random perturbation
            dx = np.random.normal(0, step_size, size=n)
            x_new = x + dx
            area_new = _compute_area(x_new, n, 20000)
            
            if area_new < current_area or np.random.random() < np.exp(-(area_new - current_area) / T):
                x = x_new
                current_area = area_new
                if current_area < best_area:
                    best_area = current_area
                    best_x = x.copy()
            
            T *= cooling
            step_size *= 0.9999
        
        # Final polish
        elapsed = time.time() - start_time
        if elapsed < deadline - safety_margin:
            try:
                result = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < best_area:
                    best_area = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_x)):
        # Fallback to Keich construction
        j_arr = np.arange(1, k + 1)
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            best_x[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
