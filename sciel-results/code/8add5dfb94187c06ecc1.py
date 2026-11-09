# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n  # shape (M,)
    # L_i = x_i + a_i * y, shape (M, n)
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _area_and_grad(x, n=64, n_heights=20000):
    """Compute area and gradient using automatic differentiation via finite differences
    or analytic gradient. For speed, use a smaller number of heights for gradient."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    area = np.mean(union_lens)
    
    # Gradient: d(area)/d(x_i)
    # For each height, the union length is a piecewise linear function of x_i.
    # We use a numerical gradient with small step for robustness.
    grad = np.zeros(n)
    step = 1e-7
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += step
        L_plus = x_plus[None, :] + a[None, :] * y_vals[:, None]
        L_sorted_plus = np.sort(L_plus, axis=1)
        gaps_plus = np.diff(L_sorted_plus, axis=1)
        union_lens_plus = w_vals + np.sum(np.minimum(gaps_plus, w_vals[:, None]), axis=1)
        area_plus = np.mean(union_lens_plus)
        grad[i] = (area_plus - area) / step
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 580  # Safety margin
    
    # Start from the Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Use Nelder-Mead with a reasonable number of heights for speed
    n_heights_nm = 5000  # Fewer heights for optimization speed
    
    def objective(x):
        a = (np.arange(n) + 1) / n
        y_vals = (np.arange(n_heights_nm) + 0.5) / n_heights_nm
        w_vals = (1.0 - y_vals) / n
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
        return np.mean(union_lens)
    
    # Try multiple random restarts with Nelder-Mead
    np.random.seed(42)
    
    # First try from the Keich starting point
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    if result.fun < best_area:
        best_area = result.fun
        best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        # Random perturbation from best
        x_start = best_x + np.random.normal(0, 0.01, size=n)
        result = minimize(
            objective, x_start, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Final refinement with more heights
    if time.time() < deadline - 60:
        def objective_full(x):
            a = (np.arange(n) + 1) / n
            y_vals = (np.arange(n_heights_nm) + 0.5) / n_heights_nm
            w_vals = (1.0 - y_vals) / n
            L = x[None, :] + a[None, :] * y_vals[:, None]
            L_sorted = np.sort(L, axis=1)
            gaps = np.diff(L_sorted, axis=1)
            union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
            return np.mean(union_lens)
        
        result = minimize(
            objective_full, best_x, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
