# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_area(x):
    """Compute the area of the union of 64 triangles given base positions x."""
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    # Heights: y_m = (m + 0.5) / 20000, m = 0..19999
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0  # shape (20000,)
    
    # For each triangle i, L_i = x_i + a_i * y, w = (1 - y) / 64
    # L has shape (20000, 64): L[m, i] = x[i] + a[i] * y[m]
    # w has shape (20000,): w[m] = (1 - y[m]) / 64
    
    # Vectorized: L = x[None, :] + a[None, :] * y[:, None]
    L = x[None, :] + a[None, :] * y[:, None]  # shape (20000, 64)
    w = (1.0 - y) / 64.0  # shape (20000,)
    
    # Sort L along axis=1
    L_sorted = np.sort(L, axis=1)  # shape (20000, 64)
    
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    # Consecutive differences: d_j = L_sorted[:, j+1] - L_sorted[:, j] for j=0..62
    d = np.diff(L_sorted, axis=1)  # shape (20000, 63)
    
    # min(d_j, w) for each pair
    d_clipped = np.minimum(d, w[:, None])  # shape (20000, 63)
    
    # Union length at each height
    union_lengths = w + d_clipped.sum(axis=1)  # shape (20000,)
    
    # Area = mean of union lengths
    area = union_lengths.mean()
    
    return area


def _compute_area_grad(x):
    """Compute gradient of area with respect to x using central differences."""
    n = 64
    eps = 1e-7
    grad = np.zeros(n)
    base_area = _compute_area(x)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_compute_area(x_plus) - _compute_area(x_minus)) / (2 * eps)
    
    return grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses L-BFGS-B optimization starting from Keich construction.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s safety margin
    
    n = 64
    
    # Initial Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    # Try optimization
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Use L-BFGS-B with numerical gradients
    # First, try with a reasonable number of iterations
    try:
        result = minimize(
            _compute_area,
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 10000,
                'maxls': 50
            }
        )
        
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
            
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Try a second round of optimization from the best point found
    try:
        result2 = minimize(
            _compute_area,
            best_x,
            method='L-BFGS-B',
            options={
                'maxiter': 300,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxfun': 5000,
                'maxls': 50
            }
        )
        
        if result2.fun < best_area:
            best_area = result2.fun
            best_x = result2.x.copy()
            
    except Exception:
        pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_x
    
    # Return the best solution found
    return best_x


# EVOLVE-BLOCK-END
