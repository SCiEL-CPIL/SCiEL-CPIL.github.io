# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses separable construction: x_k depends only on i, y_k depends only on j.
    Optimizes the 1D base positions using L-BFGS-B on the analytic integral
    of the union length over c.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner), column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    L = 1.0 / N  # tube cross-section size
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    
    def compute_1d_union_length_sq(x, c):
        """
        Compute the squared union length of intervals at height c.
        x: array of N base positions
        c: scalar in [0, 1]
        Returns: union length (sum of lengths of merged intervals)
        """
        # At height c, tube i occupies [x_i + c*i/N, x_i + c*i/N + L]
        shifted = x + c * np.arange(N) / N
        # Sort by left endpoint
        order = np.argsort(shifted)
        sorted_left = shifted[order]
        sorted_right = sorted_left + L
        
        # Merge intervals
        merged_left = [sorted_left[0]]
        merged_right = [sorted_right[0]]
        for k in range(1, N):
            if sorted_left[k] <= merged_right[-1]:
                merged_right[-1] = max(merged_right[-1], sorted_right[k])
            else:
                merged_left.append(sorted_left[k])
                merged_right.append(sorted_right[k])
        
        union_len = sum(r - l for l, r in zip(merged_left, merged_right))
        return union_len
    
    def compute_1d_cost(x):
        """
        Compute the integral over c in [0,1] of the union length.
        This is the 1D 'volume' (area in x-c plane).
        We use numerical integration with many samples for accuracy.
        """
        # Use adaptive quadrature-like approach with many points
        c_samples = np.linspace(0, 1, 200)
        total = 0.0
        for c in c_samples:
            total += compute_1d_union_length_sq(x, c)
        # Trapezoidal rule
        cost = (total - 0.5 * compute_1d_union_length_sq(x, 0) - 0.5 * compute_1d_union_length_sq(x, 1)) / 199
        return cost
    
    def compute_1d_cost_grad(x):
        """
        Numerical gradient of the 1D cost.
        """
        eps = 1e-7
        grad = np.zeros(N)
        f0 = compute_1d_cost(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            fp = compute_1d_cost(xp)
            xm = x.copy()
            xm[i] -= eps
            fm = compute_1d_cost(xm)
            grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    # Initial guess: the separable construction from the seed
    x_1d_init = -np.arange(N) / (2.0 * N)
    
    # Optimize x_1d
    best_x = x_1d_init.copy()
    best_cost = compute_1d_cost(x_1d_init)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            compute_1d_cost,
            x_1d_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success and result.fun < best_cost:
            best_x = result.x
            best_cost = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for y optimization, use same as x
        y_1d = best_x.copy()
    else:
        # Optimize y_1d similarly
        y_1d_init = -np.arange(N) / (2.0 * N)
        best_y = y_1d_init.copy()
        best_cost_y = compute_1d_cost(y_1d_init)
        
        try:
            result_y = minimize(
                compute_1d_cost,
                y_1d_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_y.success and result_y.fun < best_cost_y:
                best_y = result_y.x
                best_cost_y = result_y.fun
        except Exception:
            pass
        y_1d = best_y
    
    # Construct the 64 positions
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    base_x = best_x[i_idx]
    base_y = y_1d[j_idx]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
