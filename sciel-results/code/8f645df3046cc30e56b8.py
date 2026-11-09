# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses numerical optimization to minimize the union area of 64 triangles.
    """
    n = 64
    k = int(np.log2(n))
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Use a subset of heights for faster evaluation during optimization
    # Full evaluation uses 20000 heights, but for optimization we can use fewer
    # and verify at the end
    m_subset = np.arange(0, 20000, 20)  # 1000 heights for optimization
    y_subset = (m_subset + 0.5) / 20000.0
    
    def area_objective(x):
        """Compute union area using subset of heights for speed."""
        a = (np.arange(n) + 1) / n
        # L_i = x_i + a_i * y for each y in y_subset
        # Shape: (len(y_subset), n)
        L = x[None, :] + a[None, :] * y_subset[:, None]
        
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        
        # w at each y
        w = (1 - y_subset) / n  # shape (len(y_subset),)
        
        # Consecutive differences
        diffs = np.diff(L_sorted, axis=1)  # shape (len(y_subset), n-1)
        
        # Union length = w + sum(min(diffs, w))
        # min(diffs, w) where w is broadcast
        mins = np.minimum(diffs, w[:, None])
        union_lengths = w + np.sum(mins, axis=1)
        
        return np.mean(union_lengths)
    
    # Optimize using Powell's method
    result = minimize(
        area_objective,
        x0,
        method='Powell',
        options={
            'maxiter': 500,
            'maxfev': 50000,
            'xtol': 1e-10,
            'ftol': 1e-10
        }
    )
    
    x_opt = result.x
    
    # Verify with full 20000 heights
    m_full = np.arange(20000)
    y_full = (m_full + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    L_full = x_opt[None, :] + a[None, :] * y_full[:, None]
    L_sorted = np.sort(L_full, axis=1)
    w_full = (1 - y_full) / n
    diffs_full = np.diff(L_sorted, axis=1)
    mins_full = np.minimum(diffs_full, w_full[:, None])
    union_lengths_full = w_full + np.sum(mins_full, axis=1)
    final_area = np.mean(union_lengths_full)
    
    # Sanity check: if area is invalid, fall back to initial guess
    if not np.isfinite(final_area) or final_area <= 0:
        return x0
    
    return x_opt.astype(np.float64)


# EVOLVE-BLOCK-END
