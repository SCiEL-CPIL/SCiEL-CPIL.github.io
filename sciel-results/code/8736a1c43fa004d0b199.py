# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 600  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Define objective function with configurable resolution
    def make_objective(n_heights):
        a = (np.arange(n) + 1) / n
        y_vals = (np.arange(n_heights) + 0.5) / n_heights
        w_vals = (1.0 - y_vals) / n
        def objective(x):
            L = x[None, :] + a[None, :] * y_vals[:, None]
            L_sorted = np.sort(L, axis=1)
            gaps = np.diff(L_sorted, axis=1)
            union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
            return np.mean(union_lens)
        return objective

    # Strategy 1: Nelder-Mead with medium resolution (fast)
    objective_fast = make_objective(8000)
    np.random.seed(42)
    
    result = minimize(
        objective_fast, x0, method='Nelder-Mead',
        options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    if result.fun < best_area:
        best_area = result.fun
        best_x = result.x.copy()
    
    if time.time() > deadline - 100:
        return best_x.astype(np.float64)

    # Strategy 2: Powell's Method (derivative-free, robust for non-smooth)
    # Powell's method is often better than Nelder-Mead for this type of problem
    # as it searches along axes and can handle the piecewise linear nature well.
    objective_medium = make_objective(10000)
    try:
        result = minimize(
            objective_medium, best_x, method='Powell',
            options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 100:
        return best_x.astype(np.float64)

    # Strategy 3: Nelder-Mead restarts with perturbations
    for restart in range(5):
        if time.time() > deadline - 100:
            break
        scale = 0.005 * (1 + restart * 0.3)
        x_start = best_x + np.random.normal(0, scale, size=n)
        result = minimize(
            objective_fast, x_start, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()

    if time.time() > deadline - 50:
        return best_x.astype(np.float64)

    # Strategy 4: Final polish with high resolution Nelder-Mead
    objective_high = make_objective(20000)
    result = minimize(
        objective_high, best_x, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
    )
    if result.fun < best_area:
        best_area = result.fun
        best_x = result.x.copy()

    # Verify with full precision
    final_area = _compute_area(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        # Fallback to Keich
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
